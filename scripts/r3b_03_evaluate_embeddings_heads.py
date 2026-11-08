import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

from r3b_common import (
    BASE_DIR,
    PROJECT_ROOT,
    load_gtzan_metadata,
    load_pmemo_metadata,
)

GENRE_GRID = {"clf__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0]}
VA_GRID = {"reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}

def evaluate_gtzan_model(model_name: str, X_df: pd.DataFrame, df_meta: pd.DataFrame, folds: list, is_mert: bool = False, X_df_alt: pd.DataFrame = None):
    print(f"\n--- Evaluating GTZAN: {model_name} (15 artist-disjoint outer folds) ---")
    results = []
    
    y_all = df_meta["label"]
    artists_all = df_meta["artist_id"]
    
    for fold_info in folds:
        rep = fold_info["repeat"]
        f_idx = fold_info["fold"]
        inner_seed = 42 + rep * 10 + f_idx
        
        train_idx = fold_info["train_tracks"]
        test_idx = fold_info["test_tracks"]
        
        y_train, y_test = y_all.loc[train_idx], y_all.loc[test_idx]
        groups_train = artists_all.loc[train_idx]
        
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=False)
        
        if not is_mert:
            X_tr, X_te = X_df.loc[train_idx].values, X_df.loc[test_idx].values
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=inner_seed))
            ])
            search = RandomizedSearchCV(
                pipe,
                GENRE_GRID,
                n_iter=8,
                cv=inner_cv,
                scoring="accuracy",
                random_state=inner_seed,
                n_jobs=1,
            )
            search.fit(X_tr, y_train, groups=groups_train)
            best_model = search.best_estimator_
            selected_variant = "standard"
            best_params = search.best_params_
        else:
            # MERT: evaluate last vs layeravg inside inner CV
            X_tr_l, X_te_l = X_df.loc[train_idx].values, X_df.loc[test_idx].values
            X_tr_a, X_te_a = X_df_alt.loc[train_idx].values, X_df_alt.loc[test_idx].values
            
            pipe_l = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=inner_seed))
            ])
            search_l = RandomizedSearchCV(pipe_l, GENRE_GRID, n_iter=8, cv=inner_cv, scoring="accuracy", random_state=inner_seed, n_jobs=1)
            search_l.fit(X_tr_l, y_train, groups=groups_train)
            
            pipe_a = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=inner_seed))
            ])
            search_a = RandomizedSearchCV(pipe_a, GENRE_GRID, n_iter=8, cv=inner_cv, scoring="accuracy", random_state=inner_seed, n_jobs=1)
            search_a.fit(X_tr_a, y_train, groups=groups_train)
            
            if search_a.best_score_ > search_l.best_score_:
                best_model = search_a.best_estimator_
                X_te = X_te_a
                selected_variant = "layeravg"
                best_params = search_a.best_params_
            else:
                best_model = search_l.best_estimator_
                X_te = X_te_l
                selected_variant = "last"
                best_params = search_l.best_params_
                
        preds = best_model.predict(X_te)
        acc = accuracy_score(y_test, preds)
        mf1 = f1_score(y_test, preds, average="macro")
        
        results.append({
            "model": model_name,
            "repeat": rep,
            "fold": f_idx,
            "accuracy": acc,
            "macro_f1": mf1,
            "selected_variant": selected_variant,
            "best_params": str(best_params),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        print(f"  [{model_name}] Rep {rep} Fold {f_idx}: Acc={acc:.4f}, Macro-F1={mf1:.4f} ({selected_variant})")
        
    df_res = pd.DataFrame(results)
    print(f"[{model_name}] Mean Accuracy: {df_res['accuracy'].mean():.4f} +/- {df_res['accuracy'].std():.4f}, Macro-F1: {df_res['macro_f1'].mean():.4f} +/- {df_res['macro_f1'].std():.4f}")
    return df_res


def evaluate_pmemo_model(model_name: str, X_df: pd.DataFrame, df_meta: pd.DataFrame, folds: list, is_mert: bool = False, X_df_alt: pd.DataFrame = None):
    print(f"\n--- Evaluating PMEmo: {model_name} (10 artist-grouped outer folds) ---")
    results = []
    
    artists_all = df_meta["artist_id"]
    targets = ["valence", "arousal"]
    
    for fold_info in folds:
        rep = fold_info["repeat"]
        f_idx = fold_info["fold"]
        inner_seed = 42 + rep * 10 + f_idx
        
        train_idx = [str(x) for x in fold_info["train_tracks"]]
        test_idx = [str(x) for x in fold_info["test_tracks"]]
        
        groups_train = artists_all.loc[train_idx]
        inner_cv = GroupKFold(n_splits=3)
        
        for tgt in targets:
            col_name = "Valence(mean)" if tgt == "valence" else "Arousal(mean)"
            y_all = df_meta[col_name]
            y_train, y_test = y_all.loc[train_idx], y_all.loc[test_idx]
            
            if not is_mert:
                X_tr, X_te = X_df.loc[train_idx].values, X_df.loc[test_idx].values
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge())
                ])
                search = RandomizedSearchCV(pipe, VA_GRID, n_iter=8, cv=inner_cv, scoring="r2", random_state=inner_seed, n_jobs=1)
                search.fit(X_tr, y_train, groups=groups_train)
                best_model = search.best_estimator_
                selected_variant = "standard"
                best_params = search.best_params_
            else:
                X_tr_l, X_te_l = X_df.loc[train_idx].values, X_df.loc[test_idx].values
                X_tr_a, X_te_a = X_df_alt.loc[train_idx].values, X_df_alt.loc[test_idx].values
                
                pipe_l = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
                search_l = RandomizedSearchCV(pipe_l, VA_GRID, n_iter=8, cv=inner_cv, scoring="r2", random_state=inner_seed, n_jobs=1)
                search_l.fit(X_tr_l, y_train, groups=groups_train)
                
                pipe_a = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
                search_a = RandomizedSearchCV(pipe_a, VA_GRID, n_iter=8, cv=inner_cv, scoring="r2", random_state=inner_seed, n_jobs=1)
                search_a.fit(X_tr_a, y_train, groups=groups_train)
                
                if search_a.best_score_ > search_l.best_score_:
                    best_model = search_a.best_estimator_
                    X_te = X_te_a
                    selected_variant = "layeravg"
                    best_params = search_a.best_params_
                else:
                    best_model = search_l.best_estimator_
                    X_te = X_te_l
                    selected_variant = "last"
                    best_params = search_l.best_params_
                    
            preds = best_model.predict(X_te)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            results.append({
                "model": model_name,
                "target": tgt,
                "repeat": rep,
                "fold": f_idx,
                "r2": r2,
                "mae": mae,
                "selected_variant": selected_variant,
                "best_params": str(best_params),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })
            print(f"  [{model_name}|{tgt}] Rep {rep} Fold {f_idx}: R2={r2:.4f}, MAE={mae:.4f} ({selected_variant})")
            
    df_res = pd.DataFrame(results)
    for tgt in targets:
        sub = df_res[df_res["target"] == tgt]
        print(f"[{model_name}|{tgt}] Mean R2: {sub['r2'].mean():.4f} +/- {sub['r2'].std():.4f}, MAE: {sub['mae'].mean():.4f} +/- {sub['mae'].std():.4f}")
    return df_res


def main():
    print("=== Step 3: Evaluate Embedding Baselines with Tuned Linear Heads ===")
    t_start = time.perf_counter()
    
    df_g_meta, folds_g = load_gtzan_metadata()
    df_p_meta, folds_p = load_pmemo_metadata()
    
    # ---------------------------------------------------------
    # 1. Load Embeddings
    # ---------------------------------------------------------
    # VGGish
    vggish_g_csv = os.path.join(PROJECT_ROOT, "experiments", "rev2_p5_deep", "vggish_gtzan_embeddings.csv")
    vggish_p_csv = os.path.join(PROJECT_ROOT, "experiments", "rev2_p5_deep", "vggish_pmemo_embeddings.csv")
    df_vgg_g = pd.read_csv(vggish_g_csv).set_index("track_id")
    df_vgg_p = pd.read_csv(vggish_p_csv)
    df_vgg_p["track_id"] = df_vgg_p["track_id"].astype(str)
    df_vgg_p = df_vgg_p.set_index("track_id")
    
    # PANNs CNN14
    panns_g_csv = os.path.join(BASE_DIR, "panns_gtzan_embeddings.csv")
    panns_p_csv = os.path.join(BASE_DIR, "panns_pmemo_embeddings.csv")
    assert os.path.isfile(panns_g_csv), f"Missing {panns_g_csv}"
    assert os.path.isfile(panns_p_csv), f"Missing {panns_p_csv}"
    df_pan_g = pd.read_csv(panns_g_csv).set_index("track_id")
    df_pan_p = pd.read_csv(panns_p_csv)
    df_pan_p["track_id"] = df_pan_p["track_id"].astype(str)
    df_pan_p = df_pan_p.set_index("track_id")
    
    # MERT-v1-95M
    mert_g_last_csv = os.path.join(BASE_DIR, "mert_gtzan_last_embeddings.csv")
    mert_g_avg_csv = os.path.join(BASE_DIR, "mert_gtzan_layeravg_embeddings.csv")
    mert_p_last_csv = os.path.join(BASE_DIR, "mert_pmemo_last_embeddings.csv")
    mert_p_avg_csv = os.path.join(BASE_DIR, "mert_pmemo_layeravg_embeddings.csv")
    assert os.path.isfile(mert_g_last_csv), f"Missing {mert_g_last_csv}"
    assert os.path.isfile(mert_g_avg_csv), f"Missing {mert_g_avg_csv}"
    assert os.path.isfile(mert_p_last_csv), f"Missing {mert_p_last_csv}"
    assert os.path.isfile(mert_p_avg_csv), f"Missing {mert_p_avg_csv}"
    
    df_mert_g_l = pd.read_csv(mert_g_last_csv).set_index("track_id")
    df_mert_g_a = pd.read_csv(mert_g_avg_csv).set_index("track_id")
    df_mert_p_l = pd.read_csv(mert_p_last_csv)
    df_mert_p_l["track_id"] = df_mert_p_l["track_id"].astype(str)
    df_mert_p_l = df_mert_p_l.set_index("track_id")
    df_mert_p_a = pd.read_csv(mert_p_avg_csv)
    df_mert_p_a["track_id"] = df_mert_p_a["track_id"].astype(str)
    df_mert_p_a = df_mert_p_a.set_index("track_id")
    
    # ---------------------------------------------------------
    # 2. Run Evaluations
    # ---------------------------------------------------------
    gtzan_results = []
    pmemo_results = []
    
    # VGGish
    gtzan_results.append(evaluate_gtzan_model("VGGish", df_vgg_g, df_g_meta, folds_g))
    pmemo_results.append(evaluate_pmemo_model("VGGish", df_vgg_p, df_p_meta, folds_p))
    
    # PANNs CNN14
    gtzan_results.append(evaluate_gtzan_model("PANNs-CNN14", df_pan_g, df_g_meta, folds_g))
    pmemo_results.append(evaluate_pmemo_model("PANNs-CNN14", df_pan_p, df_p_meta, folds_p))
    
    # MERT-v1-95M (with inner variant selection)
    gtzan_results.append(evaluate_gtzan_model("MERT-v1-95M", df_mert_g_l, df_g_meta, folds_g, is_mert=True, X_df_alt=df_mert_g_a))
    pmemo_results.append(evaluate_pmemo_model("MERT-v1-95M", df_mert_p_l, df_p_meta, folds_p, is_mert=True, X_df_alt=df_mert_p_a))
    
    # ---------------------------------------------------------
    # 3. Save Combined Results
    # ---------------------------------------------------------
    df_gtzan_all = pd.concat(gtzan_results, ignore_index=True)
    df_pmemo_all = pd.concat(pmemo_results, ignore_index=True)
    
    gtzan_csv_out = os.path.join(BASE_DIR, "gtzan_r3b_folds.csv")
    pmemo_csv_out = os.path.join(BASE_DIR, "pmemo_r3b_folds.csv")
    df_gtzan_all.to_csv(gtzan_csv_out, index=False)
    df_pmemo_all.to_csv(pmemo_csv_out, index=False)
    
    print(f"\nSaved GTZAN folds to {gtzan_csv_out}")
    print(f"Saved PMEmo folds to {pmemo_csv_out}")
    print(f"Completed evaluation in {(time.perf_counter() - t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()

