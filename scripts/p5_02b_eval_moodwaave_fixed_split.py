"""
experiments/rev2_p5_deep/02b_eval_moodwaave_fixed_split.py
Evaluates default MOODWAAVE-Lite (71 features) and MOODWAAVE-Full (113 features)
on the exact same P1 fixed Kereliuk-Sturm split (train 442, test 290)
to ensure direct head-to-head comparability with VGGish and SmallMelCNN.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common

def main():
    out_dir = common.BASE_DIR
    
    with open(common.SPLITS_JSON, "r", encoding="utf-8") as f:
        ks_split = json.load(f)["gtzan"]["kereliuk_sturm_fixed"]
        
    df_g = pd.read_csv(common.GTZAN_FEATURES_CSV).set_index("filename")
    
    # Load feature sets
    with open(os.path.join(common.PROJECT_ROOT, "experiments", "idea2_cost_pareto", "feature_groups.json")) as f:
        fg = json.load(f)["groups"]
        
    full_cols = [c for grp in ["MFCC", "Chroma", "Tonnetz", "Spectral", "SpecContrast", "Temporal"] for c in fg[grp]]
    lite_cols = [c for grp in ["MFCC", "SpecContrast", "Temporal"] for c in fg[grp]]
    
    feat_sets = {
        "MOODWAAVE-Full": full_cols,
        "MOODWAAVE-Lite": lite_cols,
    }
    
    tr_ids = ks_split["train_tracks"]
    te_ids = ks_split["test_tracks"]
    
    train_df = df_g.loc[tr_ids]
    test_df = df_g.loc[te_ids]
    
    y_tr = train_df["label"].values
    y_te = test_df["label"].values
    
    records = []
    
    for model_name, feats in feat_sets.items():
        X_tr = train_df[feats].values
        X_te = test_df[feats].values
        
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="macro")
        
        print(f"{model_name} on Fixed Split (290 test tracks): Accuracy = {acc*100:.2f}%, Macro-F1 = {f1:.4f}")
        records.append({
            "model": model_name,
            "split": "kereliuk_sturm_fixed",
            "n_train": len(tr_ids),
            "n_test": len(te_ids),
            "accuracy": float(acc),
            "macro_f1": float(f1),
        })
        
    # Check if VGGish fixed split results exist and merge
    vggish_fixed_csv = os.path.join(out_dir, "vggish_fixed_split.csv")
    if os.path.exists(vggish_fixed_csv):
        df_v = pd.read_csv(vggish_fixed_csv)
        records.append({
            "model": "VGGish + Logistic Regression",
            "split": "kereliuk_sturm_fixed",
            "n_train": int(df_v["n_train"].values[0]),
            "n_test": int(df_v["n_test"].values[0]),
            "accuracy": float(df_v["accuracy"].values[0]),
            "macro_f1": float(df_v["macro_f1"].values[0]),
        })
        
    # Check if SmallMelCNN results exist and merge
    cnn_seeds_csv = os.path.join(out_dir, "cnn_seeds_results.csv")
    if os.path.exists(cnn_seeds_csv):
        df_c = pd.read_csv(cnn_seeds_csv)
        records.append({
            "model": "SmallMelCNN (scratch)",
            "split": "kereliuk_sturm_fixed",
            "n_train": len(tr_ids),
            "n_test": len(te_ids),
            "accuracy": float(df_c["test_accuracy"].mean()),
            "macro_f1": float(df_c["test_macro_f1"].mean()),
        })
        
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "fixed_split_comparison.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved fixed split comparison to {out_csv}")

if __name__ == "__main__":
    main()

