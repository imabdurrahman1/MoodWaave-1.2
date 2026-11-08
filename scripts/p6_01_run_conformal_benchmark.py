"""
experiments/rev2_p6_conformal/01_run_conformal_benchmark.py
Benchmark of 6 Conformal Methods across 3 Zone Grids on PMEmo Lite-71.
Strict 20 artist-disjoint 60/20/20 splits. Fixed seeds. Single-threaded.
"""

import os
# Strict single-thread settings before any numeric library imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import time
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Suppress feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

from common import (
    BASE_DIR,
    log_environment,
    load_pmemo_lite,
    get_artist_grouped_splits,
    finite_sample_conformal_quantile,
    build_g1_equal_width,
    build_g2_quantile,
    build_g3_quadrants,
    LGBM_MOOD_PARAMS,
    COVERAGE_SWEEP,
    N_SPLITS,
    SEEDS,
)

def compute_pearson_safe(x: list, y: list) -> float:
    if len(x) < 2:
        return 0.0
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x < 1e-9 or std_y < 1e-9:
        return 0.0
    r = pearsonr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)

def run_benchmark():
    start_time = time.time()
    print("=" * 80)
    print("MOODWAAVE - TASK P6: Better Conformal Methods and Zone Grids")
    print("=" * 80)
    
    # 1. Log Environment
    print("[1/5] Logging environment...")
    env = log_environment(BASE_DIR)
    print(f"Logged OS={env['os']}, CPU={env['processor']}, RAM={env['ram_gb']}GB")
    print(f"LightGBM v{env['libraries']['lightgbm']}, scikit-learn v{env['libraries']['scikit-learn']}")
    
    # 2. Load Data and Generate Splits
    print("\n[2/5] Loading PMEmo Lite-71 dataset and artist groupings...")
    merged, X, y_v, y_a, artists, lite_features = load_pmemo_lite()
    print(f"Loaded {len(merged)} tracks, {len(lite_features)} Lite features, {len(np.unique(artists))} unique artists.")
    
    splits = get_artist_grouped_splits(artists, n_splits=N_SPLITS, seeds=SEEDS)
    print(f"Generated {len(splits)} artist-disjoint 60/20/20 train/cal/test splits.")
    
    # Containers for results
    point_results = []
    conformal_raw = []
    constant_k_raw = []
    
    # 3. Iterate over Splits
    print("\n[3/5] Running benchmark across 20 splits...")
    for s_idx, split in enumerate(splits):
        seed = split["seed"]
        tr_idx, cal_idx, te_idx = split["train_idx"], split["cal_idx"], split["test_idx"]
        
        # Verify 0 artist overlap
        artists_tr = set(artists[tr_idx])
        artists_cal = set(artists[cal_idx])
        artists_te = set(artists[te_idx])
        assert len(artists_tr.intersection(artists_cal)) == 0, f"Leakage tr-cal in split {s_idx}"
        assert len(artists_tr.intersection(artists_te)) == 0, f"Leakage tr-te in split {s_idx}"
        assert len(artists_cal.intersection(artists_te)) == 0, f"Leakage cal-te in split {s_idx}"
        
        X_tr, y_v_tr, y_a_tr = X[tr_idx], y_v[tr_idx], y_a[tr_idx]
        X_cal, y_v_cal, y_a_cal = X[cal_idx], y_v[cal_idx], y_a[cal_idx]
        X_te, y_v_te, y_a_te = X[te_idx], y_v[te_idx], y_a[te_idx]
        
        # Fit Point Models
        m_v = LGBMRegressor(**LGBM_MOOD_PARAMS).fit(X_tr, y_v_tr)
        m_a = LGBMRegressor(**LGBM_MOOD_PARAMS).fit(X_tr, y_a_tr)
        
        # Test predictions & metrics
        v_pred_te = m_v.predict(X_te)
        a_pred_te = m_a.predict(X_te)
        v_r2 = r2_score(y_v_te, v_pred_te)
        v_mae = mean_absolute_error(y_v_te, v_pred_te)
        a_r2 = r2_score(y_a_te, a_pred_te)
        a_mae = mean_absolute_error(y_a_te, a_pred_te)
        
        point_results.append({
            "split": s_idx,
            "seed": seed,
            "n_train": len(tr_idx),
            "n_cal": len(cal_idx),
            "n_test": len(te_idx),
            "valence_r2": float(v_r2),
            "valence_mae": float(v_mae),
            "arousal_r2": float(a_r2),
            "arousal_mae": float(a_mae),
        })
        
        # Test point Euclidean errors
        e_te = np.sqrt((y_v_te - v_pred_te)**2 + (y_a_te - a_pred_te)**2)
        
        # Cal predictions
        v_pred_cal = m_v.predict(X_cal)
        a_pred_cal = m_a.predict(X_cal)
        res_v_cal = np.abs(y_v_cal - v_pred_cal)
        res_a_cal = np.abs(y_a_cal - a_pred_cal)
        
        # Precompute M3: Dispersion models via 5-fold CV on train
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_v = np.zeros(len(tr_idx))
        oof_a = np.zeros(len(tr_idx))
        for tr_k, val_k in kf.split(X_tr):
            cv_mv = LGBMRegressor(**LGBM_MOOD_PARAMS).fit(X_tr[tr_k], y_v_tr[tr_k])
            cv_ma = LGBMRegressor(**LGBM_MOOD_PARAMS).fit(X_tr[tr_k], y_a_tr[tr_k])
            oof_v[val_k] = cv_mv.predict(X_tr[val_k])
            oof_a[val_k] = cv_ma.predict(X_tr[val_k])
            
        r_v_tr = np.abs(y_v_tr - oof_v)
        r_a_tr = np.abs(y_a_tr - oof_a)
        disp_v = LGBMRegressor(**LGBM_MOOD_PARAMS).fit(X_tr, r_v_tr)
        disp_a = LGBMRegressor(**LGBM_MOOD_PARAMS).fit(X_tr, r_a_tr)
        
        sig_v_cal = np.maximum(disp_v.predict(X_cal), 1e-3)
        sig_a_cal = np.maximum(disp_a.predict(X_cal), 1e-3)
        sig_v_te = np.maximum(disp_v.predict(X_te), 1e-3)
        sig_a_te = np.maximum(disp_a.predict(X_te), 1e-3)
        
        score_v_m3_cal = res_v_cal / sig_v_cal
        score_a_m3_cal = res_a_cal / sig_a_cal
        
        # Precompute M4: Joint 2-D standard deviations and scores
        v_pred_tr = m_v.predict(X_tr)
        a_pred_tr = m_a.predict(X_tr)
        s_v = max(1e-4, float(np.std(y_v_tr - v_pred_tr)))
        s_a = max(1e-4, float(np.std(y_a_tr - a_pred_tr)))
        score_m4_cal = np.sqrt(((y_v_cal - v_pred_cal) / s_v)**2 + ((y_a_cal - a_pred_cal) / s_a)**2)
        
        # Precompute M2 (CQR) models for each nominal coverage
        cqr_data = {}
        for cov in COVERAGE_SWEEP:
            alpha = 1.0 - cov
            beta = alpha / 4.0
            p_lo = LGBM_MOOD_PARAMS.copy()
            p_lo.update({"objective": "quantile", "alpha": beta})
            p_hi = LGBM_MOOD_PARAMS.copy()
            p_hi.update({"objective": "quantile", "alpha": 1.0 - beta})
            
            q_lo_v = LGBMRegressor(**p_lo).fit(X_tr, y_v_tr)
            q_hi_v = LGBMRegressor(**p_hi).fit(X_tr, y_v_tr)
            q_lo_a = LGBMRegressor(**p_lo).fit(X_tr, y_a_tr)
            q_hi_a = LGBMRegressor(**p_hi).fit(X_tr, y_a_tr)
            
            E_v_cal = np.maximum(q_lo_v.predict(X_cal) - y_v_cal, y_v_cal - q_hi_v.predict(X_cal))
            E_a_cal = np.maximum(q_lo_a.predict(X_cal) - y_a_cal, y_a_cal - q_hi_a.predict(X_cal))
            
            Q_v_m2 = finite_sample_conformal_quantile(E_v_cal, alpha / 2.0)
            Q_a_m2 = finite_sample_conformal_quantile(E_a_cal, alpha / 2.0)
            
            v_lo_te = q_lo_v.predict(X_te) - Q_v_m2
            v_hi_te = q_hi_v.predict(X_te) + Q_v_m2
            a_lo_te = q_lo_a.predict(X_te) - Q_a_m2
            a_hi_te = q_hi_a.predict(X_te) + Q_a_m2
            
            cqr_data[cov] = (v_lo_te, v_hi_te, a_lo_te, a_hi_te)
            
        # Build the 3 Grids
        # Ground Rule 4: G2 quantile edges strictly computed on training split
        grids = [
            ("G1", build_g1_equal_width()),
            ("G2", build_g2_quantile(y_v_tr, y_a_tr)),
            ("G3", build_g3_quadrants()),
        ]
        
        # Evaluate for each grid
        for grid_name, grid in grids:
            z_tr = grid.points_to_zones(y_v_tr, y_a_tr)
            z_cal = grid.points_to_zones(y_v_cal, y_a_cal)
            z_te = grid.points_to_zones(y_v_te, y_a_te)
            
            # Constant-k Baseline for this grid
            counts = Counter(z_tr)
            ranked_zones = [c[0] for c in counts.most_common()]
            for k in range(1, grid.num_zones + 1):
                top_k = set(ranked_zones[:k])
                cov_k = np.mean([z in top_k for z in z_te])
                constant_k_raw.append({
                    "split": s_idx,
                    "grid": grid_name,
                    "k": k,
                    "empirical_coverage": float(cov_k),
                })
                
            # Multiclass Classifier for M5 (LAC) and M6 (APS)
            clf = LGBMClassifier(**LGBM_MOOD_PARAMS).fit(X_tr, z_tr)
            
            P_cal_raw = clf.predict_proba(X_cal)
            P_cal = np.zeros((len(X_cal), grid.num_zones))
            for ic, c in enumerate(clf.classes_):
                P_cal[:, c] = P_cal_raw[:, ic]
                
            P_te_raw = clf.predict_proba(X_te)
            P_te = np.zeros((len(X_te), grid.num_zones))
            for ic, c in enumerate(clf.classes_):
                P_te[:, c] = P_te_raw[:, ic]
                
            # M5 Calibration Nonconformity Scores (1 - p_true)
            scores_m5_cal = 1.0 - P_cal[np.arange(len(z_cal)), z_cal]
            
            # M6 Calibration Nonconformity Scores (APS cumulative probability)
            scores_m6_cal = []
            for i in range(len(z_cal)):
                p = P_cal[i]
                order = np.argsort(-p)
                true_k = z_cal[i]
                k_star_idx = np.where(order == true_k)[0]
                k_star = int(k_star_idx[0]) if len(k_star_idx) > 0 else len(p) - 1
                scores_m6_cal.append(float(np.sum(p[order[:k_star + 1]])))
            scores_m6_cal = np.array(scores_m6_cal)
            
            # Sweep nominal coverage
            for cov in COVERAGE_SWEEP:
                alpha = 1.0 - cov
                alpha_marginal = alpha / 2.0
                
                # --- M1: Per-target Split-Conformal Box ---
                q_v_m1 = finite_sample_conformal_quantile(res_v_cal, alpha_marginal)
                q_a_m1 = finite_sample_conformal_quantile(res_a_cal, alpha_marginal)
                sets_m1 = [grid.box_to_zones(v - q_v_m1, v + q_v_m1, a - q_a_m1, a + q_a_m1)
                           for v, a in zip(v_pred_te, a_pred_te)]
                
                # --- M2: CQR Box ---
                v_lo_te, v_hi_te, a_lo_te, a_hi_te = cqr_data[cov]
                sets_m2 = [grid.box_to_zones(v1, v2, a1, a2)
                           for v1, v2, a1, a2 in zip(v_lo_te, v_hi_te, a_lo_te, a_hi_te)]
                
                # --- M3: Normalised Scores Box ---
                q_v_m3 = finite_sample_conformal_quantile(score_v_m3_cal, alpha_marginal)
                q_a_m3 = finite_sample_conformal_quantile(score_a_m3_cal, alpha_marginal)
                sets_m3 = [grid.box_to_zones(v - q_v_m3 * sv, v + q_v_m3 * sv, a - q_a_m3 * sa, a + q_a_m3 * sa)
                           for v, a, sv, sa in zip(v_pred_te, a_pred_te, sig_v_te, sig_a_te)]
                
                # --- M4: Joint 2-D Ellipse ---
                Q_m4 = finite_sample_conformal_quantile(score_m4_cal, alpha)
                sets_m4 = [grid.ellipse_to_zones(v, a, s_v, s_a, Q_m4)
                           for v, a in zip(v_pred_te, a_pred_te)]
                
                # --- M5: LAC (Least Ambiguous Classifier) ---
                Q_m5 = finite_sample_conformal_quantile(scores_m5_cal, alpha)
                sets_m5 = []
                for j in range(len(X_te)):
                    st = np.where(P_te[j] >= (1.0 - Q_m5))[0].tolist()
                    if not st:
                        st = [int(np.argmax(P_te[j]))]
                    sets_m5.append(sorted(st))
                    
                # --- M6: APS (Adaptive Prediction Sets) ---
                Q_m6 = finite_sample_conformal_quantile(scores_m6_cal, alpha)
                sets_m6 = []
                for j in range(len(X_te)):
                    p = P_te[j]
                    order = np.argsort(-p)
                    cumsum = np.cumsum(p[order])
                    idx = np.where(cumsum >= Q_m6)[0]
                    if len(idx) > 0:
                        m = idx[0]
                        st = order[:m + 1].tolist()
                    else:
                        st = order.tolist()
                    if not st:
                        st = [order[0]]
                    sets_m6.append(sorted(st))
                    
                # Evaluate metrics for M1-M6
                method_sets = {
                    "M1_Split_Conformal": sets_m1,
                    "M2_CQR": sets_m2,
                    "M3_Normalised": sets_m3,
                    "M4_Joint_2D": sets_m4,
                    "M5_LAC": sets_m5,
                    "M6_APS": sets_m6,
                }
                
                for m_name, sets in method_sets.items():
                    emp_cov = np.mean([zt in st for zt, st in zip(z_te, sets)])
                    sizes = [len(st) for st in sets]
                    mean_sz = float(np.mean(sizes))
                    sing_rate = float(np.mean([s == 1 for s in sizes]))
                    corr = compute_pearson_safe(sizes, e_te)
                    
                    conformal_raw.append({
                        "split": s_idx,
                        "grid": grid_name,
                        "method": m_name,
                        "nominal_coverage": cov,
                        "empirical_coverage": float(emp_cov),
                        "mean_set_size": mean_sz,
                        "singleton_rate": sing_rate,
                        "size_error_corr": corr,
                    })
                    
        print(f"  Split {s_idx+1:02d}/20 completed (Seed {seed}). Valence R2={v_r2:.3f}, Arousal R2={a_r2:.3f}")
        
    print(f"\n[4/5] Benchmark completed in {time.time() - start_time:.2f}s!")
    
    # 4. Save and Aggregate Results
    print("\n[5/5] Processing tables and exporting CSVs...")
    
    # A. Point Models
    df_point = pd.DataFrame(point_results)
    mean_row = {
        "split": "MEAN",
        "seed": "-",
        "n_train": df_point["n_train"].mean(),
        "n_cal": df_point["n_cal"].mean(),
        "n_test": df_point["n_test"].mean(),
        "valence_r2": df_point["valence_r2"].mean(),
        "valence_mae": df_point["valence_mae"].mean(),
        "arousal_r2": df_point["arousal_r2"].mean(),
        "arousal_mae": df_point["arousal_mae"].mean(),
    }
    sd_row = {
        "split": "SD",
        "seed": "-",
        "n_train": df_point["n_train"].std(),
        "n_cal": df_point["n_cal"].std(),
        "n_test": df_point["n_test"].std(),
        "valence_r2": df_point["valence_r2"].std(),
        "valence_mae": df_point["valence_mae"].std(),
        "arousal_r2": df_point["arousal_r2"].std(),
        "arousal_mae": df_point["arousal_mae"].std(),
    }
    df_point_summary = pd.concat([df_point, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
    df_point_summary.to_csv(os.path.join(BASE_DIR, "point_models_performance.csv"), index=False)
    print(f"Saved point_models_performance.csv")
    print(f"Point Models Mean +/- SD:")
    print(f"  Valence: R2 = {mean_row['valence_r2']:.4f} +/- {sd_row['valence_r2']:.4f}, MAE = {mean_row['valence_mae']:.4f} +/- {sd_row['valence_mae']:.4f}")
    print(f"  Arousal: R2 = {mean_row['arousal_r2']:.4f} +/- {sd_row['arousal_r2']:.4f}, MAE = {mean_row['arousal_mae']:.4f} +/- {sd_row['arousal_mae']:.4f}")
    
    # B. Conformal Sweep Raw and Summary
    df_conf_raw = pd.DataFrame(conformal_raw)
    df_conf_raw.to_csv(os.path.join(BASE_DIR, "conformal_sweep_raw.csv"), index=False)
    
    summary_grouped = df_conf_raw.groupby(["grid", "method", "nominal_coverage"]).agg(
        empirical_coverage_mean=("empirical_coverage", "mean"),
        empirical_coverage_sd=("empirical_coverage", "std"),
        mean_set_size_mean=("mean_set_size", "mean"),
        mean_set_size_sd=("mean_set_size", "std"),
        singleton_rate_mean=("singleton_rate", "mean"),
        singleton_rate_sd=("singleton_rate", "std"),
        size_error_corr_mean=("size_error_corr", "mean"),
        size_error_corr_sd=("size_error_corr", "std"),
    ).reset_index()
    summary_grouped.to_csv(os.path.join(BASE_DIR, "conformal_sweep_results.csv"), index=False)
    print(f"Saved conformal_sweep_results.csv ({len(summary_grouped)} rows)")
    
    # C. Constant-k Baseline
    df_const_raw = pd.DataFrame(constant_k_raw)
    df_const_raw.to_csv(os.path.join(BASE_DIR, "constant_k_raw.csv"), index=False)
    
    const_summary = df_const_raw.groupby(["grid", "k"]).agg(
        empirical_coverage_mean=("empirical_coverage", "mean"),
        empirical_coverage_sd=("empirical_coverage", "std"),
    ).reset_index()
    const_summary.to_csv(os.path.join(BASE_DIR, "constant_k_baseline.csv"), index=False)
    print(f"Saved constant_k_baseline.csv ({len(const_summary)} rows)")
    
    # D. Grid Comparison at 80% Nominal Coverage
    df_80 = summary_grouped[summary_grouped["nominal_coverage"] == 0.80].copy()
    df_80.to_csv(os.path.join(BASE_DIR, "grid_comparison_80pct.csv"), index=False)
    print(f"Saved grid_comparison_80pct.csv")
    
    print("\nBenchmark successfully finished!")

if __name__ == "__main__":
    run_benchmark()

