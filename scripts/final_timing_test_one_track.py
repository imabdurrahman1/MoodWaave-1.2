"""
experiments/rev2_final_timing/test_all_pipelines_one_track.py
Quick verification of all 6 pipelines on 1 track.
"""

import os
import time
import numpy as np
import pandas as pd
import librosa
import joblib
import torch

import common as C

def test_single_track():
    print("Testing 1 track across all 6 pipelines...")
    
    # 1. Load models
    MODELS_DIR = os.path.join(C.BASE_DIR, "models")
    m_orig_g = joblib.load(os.path.join(MODELS_DIR, "pipe_a_genre_orig113.pkl"))
    m_orig_v = joblib.load(os.path.join(MODELS_DIR, "pipe_a_val_orig113.pkl"))
    m_orig_a = joblib.load(os.path.join(MODELS_DIR, "pipe_a_aro_orig113.pkl"))
    
    m_v1_g = joblib.load(os.path.join(MODELS_DIR, "pipe_b_genre_full113_v1.pkl"))
    m_v1_v = joblib.load(os.path.join(MODELS_DIR, "pipe_b_val_full113_v1.pkl"))
    m_v1_a = joblib.load(os.path.join(MODELS_DIR, "pipe_b_aro_full113_v1.pkl"))
    
    m_lite_g = joblib.load(os.path.join(MODELS_DIR, "pipe_c_genre_lite71.pkl"))
    m_lite_v = joblib.load(os.path.join(MODELS_DIR, "pipe_c_val_lite71.pkl"))
    m_lite_a = joblib.load(os.path.join(MODELS_DIR, "pipe_c_aro_lite71.pkl"))
    
    m_83_g = joblib.load(os.path.join(MODELS_DIR, "pipe_d_genre_lite83_v1.pkl"))
    m_83_v = joblib.load(os.path.join(MODELS_DIR, "pipe_d_val_lite83_v1.pkl"))
    m_83_a = joblib.load(os.path.join(MODELS_DIR, "pipe_d_aro_lite83_v1.pkl"))
    
    cnn_model = C.SmallMelCNN(num_classes=10, dropout=0.2)
    cnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "small_cnn_seed_42.pt"), map_location="cpu"))
    cnn_model.eval()
    
    vggish = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    vggish.eval()
    
    vgg_g = joblib.load(os.path.join(MODELS_DIR, "vggish_genre_head.pkl"))
    vgg_v = joblib.load(os.path.join(MODELS_DIR, "vggish_val_head.pkl"))
    vgg_a = joblib.load(os.path.join(MODELS_DIR, "vggish_aro_head.pkl"))
    
    test_track = os.path.join(C.PROJECT_ROOT, "data", "gtzan", "genres_original", "blues", "blues.00000.wav")
    print(f"Test track: {test_track}")
    
    # a) Original-113
    t0 = time.perf_counter()
    y, sr = librosa.load(test_track, sr=22050, mono=True, duration=30.0)
    t1 = time.perf_counter()
    f_orig = C.extract_original_113_v0(y, sr)
    x_orig = np.array([[f_orig[c] for c in C.FULL_FEATURES_ORIG]])
    t2 = time.perf_counter()
    g_a = m_orig_g.predict(x_orig)
    v_a = m_orig_v.predict(x_orig)
    a_a = m_orig_a.predict(x_orig)
    t3 = time.perf_counter()
    print(f"Pipeline a (Original-113): Load={(t1-t0)*1000:.1f}ms, Feat={(t2-t1)*1000:.1f}ms, Pred={(t3-t2)*1000:.1f}ms, Total={(t3-t0)*1000:.1f}ms | Genre={g_a[0]}, Val={v_a[0]:.3f}, Aro={a_a[0]:.3f}")
    
    # b) Full-113 V1
    t0 = time.perf_counter()
    y, sr = librosa.load(test_track, sr=22050, mono=True, duration=30.0)
    t1 = time.perf_counter()
    f_v1 = C.extract_full_113_v1(y, sr)
    x_v1 = np.array([[f_v1[c] for c in C.FULL_FEATURES_ORIG]])
    t2 = time.perf_counter()
    g_b = m_v1_g.predict(x_v1)
    v_b = m_v1_v.predict(x_v1)
    a_b = m_v1_a.predict(x_v1)
    t3 = time.perf_counter()
    print(f"Pipeline b (Full-113 V1): Load={(t1-t0)*1000:.1f}ms, Feat={(t2-t1)*1000:.1f}ms, Pred={(t3-t2)*1000:.1f}ms, Total={(t3-t0)*1000:.1f}ms | Genre={g_b[0]}, Val={v_b[0]:.3f}, Aro={a_b[0]:.3f}")
    
    # c) MOODWAAVE-Lite-71
    t0 = time.perf_counter()
    y, sr = librosa.load(test_track, sr=22050, mono=True, duration=30.0)
    t1 = time.perf_counter()
    f_lite = C.extract_lite_71(y, sr)
    x_lite = np.array([[f_lite[c] for c in C.LITE_FEATURES]])
    t2 = time.perf_counter()
    g_c = m_lite_g.predict(x_lite)
    v_c = m_lite_v.predict(x_lite)
    a_c = m_lite_a.predict(x_lite)
    t3 = time.perf_counter()
    print(f"Pipeline c (MOODWAAVE-Lite): Load={(t1-t0)*1000:.1f}ms, Feat={(t2-t1)*1000:.1f}ms, Pred={(t3-t2)*1000:.1f}ms, Total={(t3-t0)*1000:.1f}ms | Genre={g_c[0]}, Val={v_c[0]:.3f}, Aro={a_c[0]:.3f}")
    
    # d) Lite + V1 (83)
    t0 = time.perf_counter()
    y, sr = librosa.load(test_track, sr=22050, mono=True, duration=30.0)
    t1 = time.perf_counter()
    f_83 = C.extract_lite_plus_v1_83(y, sr)
    x_83 = np.array([[f_83[c] for c in C.LITE_PLUS_V1_FEATURES]])
    t2 = time.perf_counter()
    g_d = m_83_g.predict(x_83)
    v_d = m_83_v.predict(x_83)
    a_d = m_83_a.predict(x_83)
    t3 = time.perf_counter()
    print(f"Pipeline d (Lite + V1 83): Load={(t1-t0)*1000:.1f}ms, Feat={(t2-t1)*1000:.1f}ms, Pred={(t3-t2)*1000:.1f}ms, Total={(t3-t0)*1000:.1f}ms | Genre={g_d[0]}, Val={v_d[0]:.3f}, Aro={a_d[0]:.3f}")
    
    # e) SmallMelCNN
    t0 = time.perf_counter()
    y, sr = librosa.load(test_track, sr=22050, mono=True, duration=30.0)
    t1 = time.perf_counter()
    mel = C.compute_log_mel(y, sr=22050)
    total_frames = mel.shape[1]
    target_frames = C.CROP_FRAMES
    n_crops = max(1, total_frames // target_frames)
    crops = [mel[:, i*target_frames:(i+1)*target_frames] for i in range(n_crops) if mel[:, i*target_frames:(i+1)*target_frames].shape[1] == target_frames]
    crops_t = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1)
    t2 = time.perf_counter()
    with torch.no_grad():
        logits = cnn_model(crops_t)
        g_e = torch.argmax(logits.mean(dim=0)).item()
    t3 = time.perf_counter()
    print(f"Pipeline e (SmallMelCNN): Load={(t1-t0)*1000:.1f}ms, Feat={(t2-t1)*1000:.1f}ms, Pred={(t3-t2)*1000:.1f}ms, Total={(t3-t0)*1000:.1f}ms | Genre={g_e}")
    
    # f) VGGish + linear heads
    t0 = time.perf_counter()
    y_16, sr_16 = librosa.load(test_track, sr=16000, mono=True, duration=30.0)
    t1 = time.perf_counter()
    with torch.no_grad():
        embs = vggish(y_16, sr_16)
    if isinstance(embs, torch.Tensor):
        embs_np = embs.cpu().numpy()
    else:
        embs_np = np.asarray(embs)
    pooled = np.concatenate([np.mean(embs_np, axis=0), np.std(embs_np, axis=0)]).reshape(1, -1)
    t2 = time.perf_counter()
    g_f = vgg_g.predict(pooled)
    v_f = vgg_v.predict(pooled)
    a_f = vgg_a.predict(pooled)
    t3 = time.perf_counter()
    print(f"Pipeline f (VGGish): Load={(t1-t0)*1000:.1f}ms, Feat={(t2-t1)*1000:.1f}ms, Pred={(t3-t2)*1000:.1f}ms, Total={(t3-t0)*1000:.1f}ms | Genre={g_f[0]}, Val={v_f[0]:.3f}, Aro={a_f[0]:.3f}")

if __name__ == "__main__":
    test_single_track()
