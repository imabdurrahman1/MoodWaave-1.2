"""
experiments/rev2_final_timing/00_setup_models.py
Trains and saves the exact models for all pipelines to measure serialized disk sizes.
Records both default and tuned hyperparameters for MOODWAAVE pipelines.
"""

import os
import shutil
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from lightgbm import LGBMClassifier, LGBMRegressor

import common as C

MODELS_DIR = os.path.join(C.BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_save_all_models():
    print("=" * 80)
    print("Setting up and serializing models for all 6 pipelines...")
    print("=" * 80)
    
    # 1. Load GTZAN data
    df_gtz = pd.read_csv(C.GTZAN_CSV)
    df_gtz_v1 = pd.read_csv(C.GTZAN_TONNETZ_V1_CSV)
    
    # Merge V1 Tonnetz into GTZAN
    tonnetz_cols = C.TONNETZ_FEATURES
    non_tonnetz_cols = [c for c in C.FULL_FEATURES_ORIG if c not in tonnetz_cols]
    
    # Full V1 GTZAN
    gtz_merged = pd.merge(df_gtz, df_gtz_v1, on="filename", suffixes=("_orig", ""))
    
    y_gtz_str = df_gtz["label"].values
    classes_gtz = sorted(list(np.unique(y_gtz_str)))
    class_to_idx = {c: i for i, c in enumerate(classes_gtz)}
    y_gtz = np.array([class_to_idx[s] for s in y_gtz_str])
    
    # 2. Load PMEmo data
    df_pme = pd.read_csv(C.PMEMO_CSV)
    df_pme_v1 = pd.read_csv(C.PMEMO_TONNETZ_V1_CSV)
    df_pme["musicId_str"] = df_pme["musicId"].astype(str)
    df_pme_v1["musicId_str"] = df_pme_v1["musicId"].astype(str)
    pme_merged = pd.merge(df_pme, df_pme_v1, on="musicId_str", suffixes=("_orig", ""))
    
    y_pme_v = df_pme["Valence(mean)"].values
    y_pme_a = df_pme["Arousal(mean)"].values
    
    sizes_report = []
    
    # --------------------------------------------------------------------------
    # Pipeline a: Original-113 (Default Hyperparameters)
    # --------------------------------------------------------------------------
    print("\n[Pipeline a] Training and saving Original-113 models...")
    X_gtz_orig = df_gtz[C.FULL_FEATURES_ORIG].values
    X_pme_orig = df_pme[C.FULL_FEATURES_ORIG].values
    
    m_orig_g = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_gtz_orig, y_gtz)
    m_orig_v = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_orig, y_pme_v)
    m_orig_a = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_orig, y_pme_a)
    
    p_orig_g = os.path.join(MODELS_DIR, "pipe_a_genre_orig113.pkl")
    p_orig_v = os.path.join(MODELS_DIR, "pipe_a_val_orig113.pkl")
    p_orig_a = os.path.join(MODELS_DIR, "pipe_a_aro_orig113.pkl")
    joblib.dump(m_orig_g, p_orig_g)
    joblib.dump(m_orig_v, p_orig_v)
    joblib.dump(m_orig_a, p_orig_a)
    
    sz_orig_g = os.path.getsize(p_orig_g) / (1024.0**2)
    sz_orig_v = os.path.getsize(p_orig_v) / (1024.0**2)
    sz_orig_a = os.path.getsize(p_orig_a) / (1024.0**2)
    sz_orig_tot = sz_orig_g + sz_orig_v + sz_orig_a
    print(f"  -> Pipeline a Total Size: {sz_orig_tot:.3f} MB (Genre: {sz_orig_g:.2f} MB, Val: {sz_orig_v:.2f} MB, Aro: {sz_orig_a:.2f} MB)")
    
    sizes_report.append({
        "pipeline": "a) Original-113 (original Tonnetz)",
        "features": 113,
        "genre_model_mb": sz_orig_g,
        "valence_model_mb": sz_orig_v,
        "arousal_model_mb": sz_orig_a,
        "embedding_or_cnn_mb": 0.0,
        "total_serialized_mb": sz_orig_tot,
        "hyperparameters": "Default LightGBM (genre n_est=200, VA n_est=100, lr=0.1, leaves=31)",
    })
    
    # --------------------------------------------------------------------------
    # Pipeline b: Full-113 with Tonnetz V1
    # --------------------------------------------------------------------------
    print("\n[Pipeline b] Training and saving Full-113 with Tonnetz V1 models...")
    # Features for V1: non_tonnetz_cols + tonnetz_cols (from df_v1)
    X_gtz_v1 = gtz_merged[C.FULL_FEATURES_ORIG].values
    X_pme_v1 = pme_merged[C.FULL_FEATURES_ORIG].values
    
    m_v1_g = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_gtz_v1, y_gtz)
    m_v1_v = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_v1, y_pme_v)
    m_v1_a = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_v1, y_pme_a)
    
    p_v1_g = os.path.join(MODELS_DIR, "pipe_b_genre_full113_v1.pkl")
    p_v1_v = os.path.join(MODELS_DIR, "pipe_b_val_full113_v1.pkl")
    p_v1_a = os.path.join(MODELS_DIR, "pipe_b_aro_full113_v1.pkl")
    joblib.dump(m_v1_g, p_v1_g)
    joblib.dump(m_v1_v, p_v1_v)
    joblib.dump(m_v1_a, p_v1_a)
    
    sz_v1_g = os.path.getsize(p_v1_g) / (1024.0**2)
    sz_v1_v = os.path.getsize(p_v1_v) / (1024.0**2)
    sz_v1_a = os.path.getsize(p_v1_a) / (1024.0**2)
    sz_v1_tot = sz_v1_g + sz_v1_v + sz_v1_a
    print(f"  -> Pipeline b Total Size: {sz_v1_tot:.3f} MB")
    
    sizes_report.append({
        "pipeline": "b) Full-113 with Tonnetz V1",
        "features": 113,
        "genre_model_mb": sz_v1_g,
        "valence_model_mb": sz_v1_v,
        "arousal_model_mb": sz_v1_a,
        "embedding_or_cnn_mb": 0.0,
        "total_serialized_mb": sz_v1_tot,
        "hyperparameters": "Default LightGBM (genre n_est=200, VA n_est=100, lr=0.1, leaves=31)",
    })
    
    # --------------------------------------------------------------------------
    # Pipeline c: MOODWAAVE-Lite-71
    # --------------------------------------------------------------------------
    print("\n[Pipeline c] Training and saving MOODWAAVE-Lite-71 models...")
    X_gtz_lite = df_gtz[C.LITE_FEATURES].values
    X_pme_lite = df_pme[C.LITE_FEATURES].values
    
    m_lite_g = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_gtz_lite, y_gtz)
    m_lite_v = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_lite, y_pme_v)
    m_lite_a = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_lite, y_pme_a)
    
    p_lite_g = os.path.join(MODELS_DIR, "pipe_c_genre_lite71.pkl")
    p_lite_v = os.path.join(MODELS_DIR, "pipe_c_val_lite71.pkl")
    p_lite_a = os.path.join(MODELS_DIR, "pipe_c_aro_lite71.pkl")
    joblib.dump(m_lite_g, p_lite_g)
    joblib.dump(m_lite_v, p_lite_v)
    joblib.dump(m_lite_a, p_lite_a)
    
    sz_lite_g = os.path.getsize(p_lite_g) / (1024.0**2)
    sz_lite_v = os.path.getsize(p_lite_v) / (1024.0**2)
    sz_lite_a = os.path.getsize(p_lite_a) / (1024.0**2)
    sz_lite_tot = sz_lite_g + sz_lite_v + sz_lite_a
    print(f"  -> Pipeline c Total Size: {sz_lite_tot:.3f} MB")
    
    sizes_report.append({
        "pipeline": "c) MOODWAAVE-Lite-71",
        "features": 71,
        "genre_model_mb": sz_lite_g,
        "valence_model_mb": sz_lite_v,
        "arousal_model_mb": sz_lite_a,
        "embedding_or_cnn_mb": 0.0,
        "total_serialized_mb": sz_lite_tot,
        "hyperparameters": "Default LightGBM (genre n_est=200, VA n_est=100, lr=0.1, leaves=31)",
    })
    
    # --------------------------------------------------------------------------
    # Pipeline d: Lite + V1 (83 features)
    # --------------------------------------------------------------------------
    print("\n[Pipeline d] Training and saving Lite + V1 (83 features) models...")
    X_gtz_83 = gtz_merged[C.LITE_PLUS_V1_FEATURES].values
    X_pme_83 = pme_merged[C.LITE_PLUS_V1_FEATURES].values
    
    m_83_g = LGBMClassifier(**C.LGBM_GENRE_PARAMS).fit(X_gtz_83, y_gtz)
    m_83_v = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_83, y_pme_v)
    m_83_a = LGBMRegressor(**C.LGBM_MOOD_PARAMS).fit(X_pme_83, y_pme_a)
    
    p_83_g = os.path.join(MODELS_DIR, "pipe_d_genre_lite83_v1.pkl")
    p_83_v = os.path.join(MODELS_DIR, "pipe_d_val_lite83_v1.pkl")
    p_83_a = os.path.join(MODELS_DIR, "pipe_d_aro_lite83_v1.pkl")
    joblib.dump(m_83_g, p_83_g)
    joblib.dump(m_83_v, p_83_v)
    joblib.dump(m_83_a, p_83_a)
    
    sz_83_g = os.path.getsize(p_83_g) / (1024.0**2)
    sz_83_v = os.path.getsize(p_83_v) / (1024.0**2)
    sz_83_a = os.path.getsize(p_83_a) / (1024.0**2)
    sz_83_tot = sz_83_g + sz_83_v + sz_83_a
    print(f"  -> Pipeline d Total Size: {sz_83_tot:.3f} MB")
    
    sizes_report.append({
        "pipeline": "d) Lite + V1 (83)",
        "features": 83,
        "genre_model_mb": sz_83_g,
        "valence_model_mb": sz_83_v,
        "arousal_model_mb": sz_83_a,
        "embedding_or_cnn_mb": 0.0,
        "total_serialized_mb": sz_83_tot,
        "hyperparameters": "Default LightGBM (genre n_est=200, VA n_est=100, lr=0.1, leaves=31)",
    })
    
    # --------------------------------------------------------------------------
    # Pipeline e: SmallMelCNN (genre only)
    # --------------------------------------------------------------------------
    print("\n[Pipeline e] Checking SmallMelCNN checkpoint...")
    dst_cnn_ckpt = os.path.join(MODELS_DIR, "small_cnn_seed_42.pt")
    if os.path.exists(C.CNN_WEIGHTS_PATH):
        shutil.copyfile(C.CNN_WEIGHTS_PATH, dst_cnn_ckpt)
    sz_cnn = os.path.getsize(dst_cnn_ckpt) / (1024.0**2)
    print(f"  -> Pipeline e Total Size: {sz_cnn:.3f} MB ({sz_cnn*1024:.1f} KB)")
    
    sizes_report.append({
        "pipeline": "e) SmallMelCNN (genre only)",
        "features": "128 log-mel bins",
        "genre_model_mb": sz_cnn,
        "valence_model_mb": 0.0,
        "arousal_model_mb": 0.0,
        "embedding_or_cnn_mb": sz_cnn,
        "total_serialized_mb": sz_cnn,
        "hyperparameters": "PyTorch SmallMelCNN (98,682 params, 4 conv blocks, dropout=0.2)",
    })
    
    # --------------------------------------------------------------------------
    # Pipeline f: VGGish + linear heads (genre + valence + arousal)
    # --------------------------------------------------------------------------
    print("\n[Pipeline f] Training and saving VGGish linear heads...")
    df_vgg_gtz = pd.read_csv(C.VGGish_GTZAN_EMBS_CSV).set_index("track_id")
    df_vgg_pme = pd.read_csv(C.VGGish_PMEMO_EMBS_CSV)
    df_vgg_pme["track_id_str"] = df_vgg_pme["track_id"].astype(str)
    df_vgg_pme = df_vgg_pme.set_index("track_id_str")
    
    emb_cols = [c for c in df_vgg_gtz.columns if c.startswith("vggish_")]
    
    # Match labels
    y_gtz_dict = dict(zip(df_gtz["filename"], y_gtz))
    valid_gtz_ids = [idx for idx in df_vgg_gtz.index if idx in y_gtz_dict]
    X_vgg_gtz = df_vgg_gtz.loc[valid_gtz_ids, emb_cols].values
    y_vgg_gtz = np.array([y_gtz_dict[idx] for idx in valid_gtz_ids])
    
    # PMEmo match
    y_pme_v_dict = dict(zip(df_pme["musicId_str"], y_pme_v))
    y_pme_a_dict = dict(zip(df_pme["musicId_str"], y_pme_a))
    valid_pme_ids = [idx for idx in df_vgg_pme.index if idx in y_pme_v_dict]
    X_vgg_pme = df_vgg_pme.loc[valid_pme_ids, emb_cols].values
    y_vgg_pme_v = np.array([y_pme_v_dict[idx] for idx in valid_pme_ids])
    y_vgg_pme_a = np.array([y_pme_a_dict[idx] for idx in valid_pme_ids])
    
    head_genre = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    head_val = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42))])
    head_aro = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42))])
    
    head_genre.fit(X_vgg_gtz, y_vgg_gtz)
    head_val.fit(X_vgg_pme, y_vgg_pme_v)
    head_aro.fit(X_vgg_pme, y_vgg_pme_a)
    
    p_vgg_g = os.path.join(MODELS_DIR, "vggish_genre_head.pkl")
    p_vgg_v = os.path.join(MODELS_DIR, "vggish_val_head.pkl")
    p_vgg_a = os.path.join(MODELS_DIR, "vggish_aro_head.pkl")
    joblib.dump(head_genre, p_vgg_g)
    joblib.dump(head_val, p_vgg_v)
    joblib.dump(head_aro, p_vgg_a)
    
    vgg_net_mb = 275.20  # official checkpoint vggish-10086976.pth
    sz_vgg_g = os.path.getsize(p_vgg_g) / (1024.0**2)
    sz_vgg_v = os.path.getsize(p_vgg_v) / (1024.0**2)
    sz_vgg_a = os.path.getsize(p_vgg_a) / (1024.0**2)
    sz_vgg_tot = vgg_net_mb + sz_vgg_g + sz_vgg_v + sz_vgg_a
    print(f"  -> Pipeline f Total Size: {sz_vgg_tot:.3f} MB (VGGish net: {vgg_net_mb:.2f} MB, heads: {sz_vgg_g+sz_vgg_v+sz_vgg_a:.4f} MB)")
    
    sizes_report.append({
        "pipeline": "f) VGGish + linear heads",
        "features": "256-d pooled (128 mean + 128 std)",
        "genre_model_mb": sz_vgg_g,
        "valence_model_mb": sz_vgg_v,
        "arousal_model_mb": sz_vgg_a,
        "embedding_or_cnn_mb": vgg_net_mb,
        "total_serialized_mb": sz_vgg_tot,
        "hyperparameters": "Pretrained AudioSet VGGish + StandardScaler + LogisticRegression(C=1.0) / Ridge(alpha=1.0)",
    })
    
    # Also add tuned hyperparameters comparison row for reference
    p4_lite_g = os.path.join(C.PROJECT_ROOT, "experiments", "rev2_p4_baselines", "saved_models", "genre_lightgbm_lite_71.pkl")
    p4_lite_v = os.path.join(C.PROJECT_ROOT, "experiments", "rev2_p4_baselines", "saved_models", "valence_lightgbm_lite_71.pkl")
    p4_lite_a = os.path.join(C.PROJECT_ROOT, "experiments", "rev2_p4_baselines", "saved_models", "arousal_lightgbm_lite_71.pkl")
    if os.path.exists(p4_lite_g) and os.path.exists(p4_lite_v) and os.path.exists(p4_lite_a):
        sz_tuned_lite = (os.path.getsize(p4_lite_g) + os.path.getsize(p4_lite_v) + os.path.getsize(p4_lite_a)) / (1024.0**2)
        sizes_report.append({
            "pipeline": "c) MOODWAAVE-Lite-71 (Tuned Hyperparameters)",
            "features": 71,
            "genre_model_mb": os.path.getsize(p4_lite_g) / (1024.0**2),
            "valence_model_mb": os.path.getsize(p4_lite_v) / (1024.0**2),
            "arousal_model_mb": os.path.getsize(p4_lite_a) / (1024.0**2),
            "embedding_or_cnn_mb": 0.0,
            "total_serialized_mb": sz_tuned_lite,
            "hyperparameters": "Tuned from rev2_p4_baselines (nested CV modal parameters)",
        })
        
    df_sizes = pd.DataFrame(sizes_report)
    df_sizes.to_csv(os.path.join(C.BASE_DIR, "system_sizes.csv"), index=False)
    print("\nSaved system_sizes.csv:")
    print(df_sizes[["pipeline", "total_serialized_mb", "hyperparameters"]].to_string())

if __name__ == "__main__":
    train_and_save_all_models()
