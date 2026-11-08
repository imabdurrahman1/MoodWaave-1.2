"""
experiments/rev2_p5_deep/02_eval_vggish_heads.py
Trains and evaluates Logistic Regression (genre) and Ridge (VA) heads on cached VGGish embeddings:
- Nested CV on P1 folds (5-fold GTZAN, 5x2 PMEmo) using RandomizedSearchCV(n_iter=8, cv=3)
- Direct evaluation on the P1 fixed Kereliuk-Sturm fault-filtered split (442 train, 290 test)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def evaluate_genre_cv(df_embs: pd.DataFrame, df_labels: pd.DataFrame, splits: list):
    """Evaluates Logistic Regression head on GTZAN nested folds."""
    feat_cols = [c for c in df_embs.columns if c.startswith("vggish_")]
    records = []
    
    for fold_info in splits:
        repeat = fold_info["repeat"]
        fold_idx = fold_info["fold"]
        tr_ids = fold_info["train_tracks"]
        te_ids = fold_info["test_tracks"]
        
        tr_df = df_embs.loc[tr_ids]
        te_df = df_embs.loc[te_ids]
        
        X_tr = tr_df[feat_cols].values
        X_te = te_df[feat_cols].values
        y_tr = df_labels.loc[tr_ids, "label"].values
        y_te = df_labels.loc[te_ids, "label"].values
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        grid = {"clf__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0]}
        
        inner_seed = 42 + repeat * 100 + fold_idx
        search = RandomizedSearchCV(
            pipe, param_distributions=grid, n_iter=8, cv=3, scoring="accuracy",
            random_state=inner_seed, n_jobs=1
        )
        search.fit(X_tr, y_tr)
        preds = search.best_estimator_.predict(X_te)
        
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="macro")
        
        records.append({
            "task": "genre",
            "model": "VGGish + LogisticRegression",
            "repeat": repeat,
            "fold": fold_idx,
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "best_params": json.dumps(search.best_params_),
        })
    return pd.DataFrame(records)

def evaluate_pmemo_cv(df_embs: pd.DataFrame, df_pmemo: pd.DataFrame, splits: list):
    """Evaluates Ridge head on PMEmo nested folds."""
    feat_cols = [c for c in df_embs.columns if c.startswith("vggish_")]
    records = []
    targets = [("valence", "Valence(mean)"), ("arousal", "Arousal(mean)")]
    
    for fold_info in splits:
        repeat = fold_info["repeat"]
        fold_idx = fold_info["fold"]
        tr_ids = [str(x) for x in fold_info["train_tracks"]]
        te_ids = [str(x) for x in fold_info["test_tracks"]]
        
        tr_df = df_embs.loc[tr_ids]
        te_df = df_embs.loc[te_ids]
        
        X_tr = tr_df[feat_cols].values
        X_te = te_df[feat_cols].values
        
        for task_name, target_col in targets:
            y_tr = df_pmemo.loc[tr_ids, target_col].values
            y_te = df_pmemo.loc[te_ids, target_col].values
            
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(random_state=42)),
            ])
            grid = {"reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
            
            target_offset = 0 if task_name == "valence" else 500
            inner_seed = 42 + target_offset + repeat * 100 + fold_idx
            search = RandomizedSearchCV(
                pipe, param_distributions=grid, n_iter=8, cv=3, scoring="r2",
                random_state=inner_seed, n_jobs=1
            )
            search.fit(X_tr, y_tr)
            preds = search.best_estimator_.predict(X_te)
            
            r2 = r2_score(y_te, preds)
            mae = mean_absolute_error(y_te, preds)
            
            records.append({
                "task": task_name,
                "model": "VGGish + Ridge",
                "repeat": repeat,
                "fold": fold_idx,
                "r2": float(r2),
                "mae": float(mae),
                "best_params": json.dumps(search.best_params_),
            })
    return pd.DataFrame(records)

def evaluate_fixed_split(df_embs: pd.DataFrame, df_labels: pd.DataFrame, ks_split: dict):
    """Evaluates VGGish + Logistic Regression on the P1 fixed Kereliuk-Sturm split."""
    feat_cols = [c for c in df_embs.columns if c.startswith("vggish_")]
    tr_ids = ks_split["train_tracks"]
    te_ids = ks_split["test_tracks"]
    
    X_tr = df_embs.loc[tr_ids, feat_cols].values
    y_tr = df_labels.loc[tr_ids, "label"].values
    X_te = df_embs.loc[te_ids, feat_cols].values
    y_te = df_labels.loc[te_ids, "label"].values
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    grid = {"clf__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0]}
    
    search = RandomizedSearchCV(pipe, grid, n_iter=8, cv=3, scoring="accuracy", random_state=42, n_jobs=1)
    search.fit(X_tr, y_tr)
    preds = search.best_estimator_.predict(X_te)
    
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    
    return {
        "model": "VGGish + Logistic Regression",
        "split": "kereliuk_sturm_fixed",
        "n_train": len(tr_ids),
        "n_test": len(te_ids),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "best_params": json.dumps(search.best_params_),
    }

def main():
    out_dir = common.BASE_DIR
    gtzan_emb_csv = os.path.join(out_dir, "vggish_gtzan_embeddings.csv")
    pmemo_emb_csv = os.path.join(out_dir, "vggish_pmemo_embeddings.csv")
    
    if not os.path.exists(gtzan_emb_csv) or not os.path.exists(pmemo_emb_csv):
        print("Embeddings not found! Please run 01_extract_vggish_embeddings.py first.")
        return
        
    df_gtzan_embs = pd.read_csv(gtzan_emb_csv).set_index("track_id")
    df_pmemo_embs = pd.read_csv(pmemo_emb_csv).set_index("track_id")
    df_pmemo_embs.index = df_pmemo_embs.index.astype(str)
    
    df_gtzan_labels = pd.read_csv(common.GTZAN_FEATURES_CSV).set_index("filename")
    df_pmemo_labels = pd.read_csv(common.PMEMO_READY_CSV)
    df_pmemo_labels["musicId_str"] = df_pmemo_labels["musicId"].astype(str)
    df_pmemo_labels.set_index("musicId_str", inplace=True)
    
    with open(common.SPLITS_JSON, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
        
    g_splits = [f for f in splits_data["gtzan"]["fault_removed_artist_grouped"] if f["repeat"] == 0]
    p_splits = [f for f in splits_data["pmemo"]["artist_grouped_5x5"] if f["repeat"] in (0, 1)]
    ks_split = splits_data["gtzan"]["kereliuk_sturm_fixed"]
    
    print("Evaluating VGGish + Logistic Regression on GTZAN 5-fold CV...")
    df_g_cv = evaluate_genre_cv(df_gtzan_embs, df_gtzan_labels, g_splits)
    df_g_cv.to_csv(os.path.join(out_dir, "vggish_gtzan_folds.csv"), index=False)
    print(f"GTZAN VGGish CV Accuracy: {df_g_cv['accuracy'].mean():.4f} ± {df_g_cv['accuracy'].std():.4f}")
    
    print("Evaluating VGGish + Ridge on PMEmo 5x2 CV...")
    df_p_cv = evaluate_pmemo_cv(df_pmemo_embs, df_pmemo_labels, p_splits)
    df_p_cv.to_csv(os.path.join(out_dir, "vggish_pmemo_folds.csv"), index=False)
    for t in ["valence", "arousal"]:
        sub = df_p_cv[df_p_cv["task"] == t]
        print(f"PMEmo VGGish CV {t.capitalize()} R²: {sub['r2'].mean():.4f} ± {sub['r2'].std():.4f}")
        
    print("\nEvaluating VGGish + Logistic Regression on P1 Fixed Kereliuk-Sturm Split...")
    fixed_res = evaluate_fixed_split(df_gtzan_embs, df_gtzan_labels, ks_split)
    df_fixed = pd.DataFrame([fixed_res])
    df_fixed.to_csv(os.path.join(out_dir, "vggish_fixed_split.csv"), index=False)
    print(f"Fixed Split (290 test tracks) Accuracy: {fixed_res['accuracy']:.4f}, Macro-F1: {fixed_res['macro_f1']:.4f}")

if __name__ == "__main__":
    main()

