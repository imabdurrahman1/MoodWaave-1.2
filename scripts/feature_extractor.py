import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        
        # Core Features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        #  Spectral Features 
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        #  Harmonic Features 
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        #  Others 
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, 'item'):
            tempo = tempo.item()
            
        features = {}
        
        # MFCC (26) - Interleaved Mean and Std
        for i in range(13):
            features[f'mfcc_mean_{i+1}'] = np.mean(mfcc[i])
            features[f'mfcc_std_{i+1}'] = np.std(mfcc[i])
            
        # Delta MFCC (26) - Interleaved Mean and Std
        for i in range(13):
            features[f'delta_mfcc_mean_{i+1}'] = np.mean(delta_mfcc[i])
            features[f'delta_mfcc_std_{i+1}'] = np.std(delta_mfcc[i])
            
        # Chroma (24)
        for i in range(12):
            features[f'chroma_mean_{i+1}'] = np.mean(chroma[i])
            features[f'chroma_std_{i+1}'] = np.std(chroma[i])
            
        # Spectral Contrast (14)
        for i in range(7):
            features[f'spectral_contrast_mean_{i+1}'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_std_{i+1}'] = np.std(spectral_contrast[i])
            
        # Tonnetz (12)
        for i in range(6):
            features[f'tonnetz_mean_{i+1}'] = np.mean(tonnetz[i])
            features[f'tonnetz_std_{i+1}'] = np.std(tonnetz[i])
            
        # Spectral & Others (10 + 1 = 11)
        # Total: 26 + 26 + 24 + 14 + 12 + 11 = 113 features
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['tempo'] = float(tempo)
        
        return features
    except Exception as e:
        return None

def process_dataset(base_path, output_csv, is_gtzan=True):
    data = []
    failed_files = []
    
    if is_gtzan:
        genres = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        for genre in genres:
            genre_path = os.path.join(base_path, genre)
            files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            print(f"Extracting features for {genre}...")
            for file in tqdm(files):
                file_path = os.path.join(genre_path, file)
                feat = extract_features(file_path)
                if feat:
                    feat['label'] = genre
                    data.append(feat)
                else:
                    failed_files.append(file_path)
    else:
        # PMEmo
        files = [f for f in os.listdir(base_path) if f.endswith('.mp3')]
        print(f"Extracting features for PMEmo songs...")
        for file in tqdm(files):
            file_path = os.path.join(base_path, file)
            feat = extract_features(file_path)
            if feat:
                feat['musicId'] = int(file.split('.')[0])
                data.append(feat)
            else:
                failed_files.append(file_path)
                
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\nTotal rows saved to {output_csv}: {len(df)}")
    if failed_files:
        print(f"Failed files ({len(failed_files)})")

if __name__ == "__main__":
    # 1. Process GTZAN
    process_dataset('data/gtzan/genres_original/', 'data/gtzan_features.csv', is_gtzan=True)
    
    # 2. Process PMEmo (if audio exists)
    if os.path.exists('data/pmemo/chorus/'):
        process_dataset('data/pmemo/chorus/', 'data/pmemo_features_new.csv', is_gtzan=False)
