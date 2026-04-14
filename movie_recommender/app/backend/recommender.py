"""
recommender.py – Core recommendation logic
"""

import os, json, ast
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE, '..', '..'))

MODEL_DIR = os.path.join(ROOT, 'models')
DATA_DIR  = os.path.join(ROOT, 'data')

# ── Load artefacts ─────────────────────────────────────────────────────────
svm_model  = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
knn_model  = joblib.load(os.path.join(MODEL_DIR, 'knn_model.pkl'))
lr_model   = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
scaler     = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
pca_model  = joblib.load(os.path.join(MODEL_DIR, 'pca_model.pkl'))
kmeans     = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
le         = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
    meta = json.load(f)

ALL_GENRES = meta['all_genres']

df = pd.read_csv(os.path.join(DATA_DIR, 'processed_movies.csv'))


# ── Helper: build feature vector from mood label ───────────────────────────
MOOD_TO_GENRE = {
    'Happy'  : 'Comedy',
    'Sad'    : 'Drama',
    'Angry'  : 'Action',
    'Relaxed': 'Romance',
    'Neutral': None,
}

MODEL_MAP = {
    'svm': svm_model,
    'knn': knn_model,
    'lr' : lr_model,
}


def mood_to_feature_vector(mood: str) -> np.ndarray:
    """Convert a mood string → scaled PCA feature vector."""
    target_genre = MOOD_TO_GENRE.get(mood)
    genre_vec = [1 if g == target_genre else 0 for g in ALL_GENRES]
    # Use median popularity/vote for a neutral numeric baseline
    num_vec = scaler.transform([[df['popularity'].median(),
                                  df['vote_average'].median()]])[0]
    feature_vec = np.array(genre_vec + list(num_vec)).reshape(1, -1)
    return pca_model.transform(feature_vec)


def recommend(mood: str, model_key: str = 'svm', top_n: int = 5):
    """
    Returns top_n movie recommendations for a given mood.

    Steps:
    1. Convert mood → predicted mood label (via chosen classifier)
    2. Filter movies whose mood_label matches
    3. Use K-Means cluster of the query to refine (pick movies from same cluster)
    4. Sort by vote_average desc and return top_n
    """
    mood = mood.strip().capitalize()
    if mood not in MOOD_TO_GENRE:
        return [], f"Unknown mood '{mood}'. Choose: Happy, Sad, Angry, Relaxed, Neutral"

    # Step 1 – Predict with classifier
    model    = MODEL_MAP.get(model_key, svm_model)
    x_pca    = mood_to_feature_vector(mood)
    pred_enc = model.predict(x_pca)[0]
    predicted_label = le.inverse_transform([pred_enc])[0]

    # Step 2 – Filter by predicted mood
    filtered = df[df['mood_label'] == predicted_label].copy()

    if filtered.empty:
        filtered = df[df['mood_label'] == mood].copy()

    # Step 3 – Cluster refinement: find the dominant cluster for this mood
    pred_cluster = kmeans.predict(x_pca)[0]
    cluster_filtered = filtered[filtered['cluster'] == pred_cluster]
    if len(cluster_filtered) >= top_n:
        filtered = cluster_filtered

    # Step 4 – Sort by rating and return top_n
    filtered = filtered.sort_values('vote_average', ascending=False).head(top_n * 3)
    # Deduplicate titles just in case
    filtered = filtered.drop_duplicates(subset='title')
    top = filtered.head(top_n)

    movies = []
    for _, row in top.iterrows():
        try:
            genres_list = ast.literal_eval(row['genres_list']) if isinstance(row['genres_list'], str) else []
        except Exception:
            genres_list = []
        movies.append({
            'title'     : str(row['title']),
            'genres'    : genres_list,
            'rating'    : round(float(row['vote_average']), 1),
            'popularity': round(float(row['popularity']), 1),
            'overview'  : str(row.get('overview', ''))[:250] + '…'
                          if len(str(row.get('overview', ''))) > 250
                          else str(row.get('overview', '')),
            'mood'      : str(row['mood_label']),
            'cluster'   : int(row['cluster']),
        })

    return movies, predicted_label


def get_metrics():
    return meta['model_metrics']


def get_metadata():
    return meta
