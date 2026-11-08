import os
import sys
import joblib
import pandas as pd
import numpy as np
import librosa
from feature_extractor import extract_features

# Global cache for models
_MODELS = {}

def load_models():
    """Load all necessary models, scalers, and encoders."""
    global _MODELS
    if not _MODELS:
        _MODELS['genre_model'] = joblib.load('models/genre_classifier.pkl')
        _MODELS['genre_scaler'] = joblib.load('models/genre_scaler.pkl')
        _MODELS['genre_encoder'] = joblib.load('models/genre_encoder.pkl')
        _MODELS['mood_model'] = joblib.load('models/mood_regressor.pkl')
        _MODELS['mood_scaler'] = joblib.load('models/mood_scaler.pkl')
    return _MODELS

def get_quadrant(v, a):
    """Determine the mood quadrant from valence and arousal scores."""
    if v >= 0.5 and a >= 0.5: return 'Happy'
    if v < 0.5 and a >= 0.5: return 'Angry'
    if v < 0.5 and a < 0.5: return 'Sad'
    return 'Calm'

def get_nuanced_mood(v, a):
    """Determine a more granular mood label based on V/A coordinates."""
    # Mapping based on typical circumplex regions
    if v >= 0.75 and a >= 0.75: return 'Excited'
    if v >= 0.75 and a >= 0.5:  return 'Happy'
    if v >= 0.5 and a >= 0.75:  return 'Energetic'
    if v >= 0.5 and a >= 0.5:   return 'Pleased'
    
    if v < 0.25 and a >= 0.75: return 'Frustrated'
    if v < 0.5 and a >= 0.75:  return 'Tense'
    if v < 0.25 and a >= 0.5:  return 'Angry'
    if v < 0.5 and a >= 0.5:
        return 'Distressed'
    
    if v < 0.25 and a < 0.25: return 'Depressed'
    if v < 0.5 and a < 0.25:  return 'Bored'
    if v < 0.25 and a < 0.5:  return 'Sad'
    if v < 0.5 and a < 0.5:   return 'Gloomy'
    
    if v >= 0.75 and a < 0.25: return 'Serene'
    if v >= 0.75 and a < 0.5:  return 'Relaxed'
    if v >= 0.5 and a < 0.25:  return 'Calm'
    if v >= 0.5 and a < 0.5:   return 'Peaceful'
    
    return 'Neutral'

def predict_song(file_path, models=None):
    """Predict genre and mood for a single audio file."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    # Load models if not provided
    if models is None:
        models = load_models()
    
    # Extract features
    feat = extract_features(file_path)
    if not feat:
        return {"error": f"Failed to extract features from: {file_path}"}
    
    # Convert to DataFrame
    X = pd.DataFrame([feat])
    
    #  Genre Prediction 
    # Ensure column order matches what scaler was trained on
    X_genre = X[models['genre_scaler'].feature_names_in_]
    X_genre_scaled = models['genre_scaler'].transform(X_genre)
    X_genre_scaled_df = pd.DataFrame(X_genre_scaled, columns=models['genre_scaler'].feature_names_in_)
    genre_idx = models['genre_model'].predict(X_genre_scaled_df)[0]
    genre_label = models['genre_encoder'].inverse_transform([genre_idx])[0]
    
    #  Mood Prediction 
    # Ensure column order matches what scaler was trained on
    X_mood = X[models['mood_scaler'].feature_names_in_]
    X_mood_scaled = models['mood_scaler'].transform(X_mood)
    X_mood_scaled_df = pd.DataFrame(X_mood_scaled, columns=models['mood_scaler'].feature_names_in_)
    mood_scores = models['mood_model'].predict(X_mood_scaled_df)[0]
    valence, arousal = mood_scores[0], mood_scores[1]
    
    quadrant = get_quadrant(valence, arousal)
    nuanced = get_nuanced_mood(valence, arousal)
    
    result = {
        "filename": os.path.basename(file_path),
        "predicted_genre": genre_label,
        "valence": round(float(valence), 4),
        "arousal": round(float(arousal), 4),
        "mood_quadrant": quadrant,
        "mood_nuanced": nuanced
    }
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <file_path>")
    else:
        file_path = sys.argv[1]
        res = predict_song(file_path)
        
        if "error" in res:
            print(f"ERROR: {res['error']}")
        else:
            print("\n" + "="*40)
            print(f"🎵 ANALYSIS RESULT: {res['filename']}")
            print("="*40)
            print(f"Genre:    {res['predicted_genre']}")
            print(f"Mood:     {res['mood_quadrant']}")
            print(f"Valence:  {res['valence']}")
            print(f"Arousal:  {res['arousal']}")
            print("="*40)
