"""
Emotion-Based Movie Recommendation System
=========================================
Complete training pipeline: preprocessing → feature engineering → 
classification (SVM, KNN, LR) → clustering (K-Means) → model saving
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from ast import literal_eval

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
VIZ_DIR    = os.path.join(os.path.dirname(__file__), 'visualizations')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,   exist_ok=True)

# ─── Colour palette (warm cinematic) ──────────────────────────────────────────
PALETTE = {
    'burgundy' : '#6B1D3A',
    'terracotta': '#C4663A',
    'cream'    : '#F7F2E8',
    'charcoal' : '#1C1A1E',
    'sage'     : '#7A9E7E',
    'gold'     : '#D4AF37',
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess():
    print("\n" + "="*60)
    print("STEP 1: Loading & Preprocessing Data")
    print("="*60)

    df = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_movies.csv'))
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Keep relevant columns
    df = df[['title', 'genres', 'popularity', 'vote_average', 'overview']].copy()

    # ── Handle missing values ──────────────────────────────────────────────────
    print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
    df['overview'].fillna('', inplace=True)
    df.dropna(subset=['title', 'genres', 'vote_average', 'popularity'], inplace=True)
    df = df[df['genres'].str.strip() != '[]']   # drop movies with no genre
    print(f"\nRows after cleaning: {len(df)}")

    # ── Parse genres JSON → list of genre names ────────────────────────────────
    def parse_genres(x):
        try:
            items = literal_eval(x)
            return [g['name'] for g in items if 'name' in g]
        except Exception:
            return []

    df['genres_list'] = df['genres'].apply(parse_genres)
    df = df[df['genres_list'].map(len) > 0]     # ensure at least one genre

    # ── Mood label mapping ─────────────────────────────────────────────────────
    MOOD_MAP = {
        'Comedy'   : 'Happy',
        'Drama'    : 'Sad',
        'Action'   : 'Angry',
        'Romance'  : 'Relaxed',
    }

    def assign_mood(genre_list):
        for g in genre_list:
            if g in MOOD_MAP:
                return MOOD_MAP[g]
        return 'Neutral'

    df['mood_label'] = df['genres_list'].apply(assign_mood)

    # ── One-hot encode genres ──────────────────────────────────────────────────
    all_genres = sorted({g for lst in df['genres_list'] for g in lst})
    for g in all_genres:
        df[f'genre_{g}'] = df['genres_list'].apply(lambda lst: int(g in lst))

    # ── Normalise numerical features ──────────────────────────────────────────
    scaler = StandardScaler()
    df[['popularity_scaled', 'vote_scaled']] = scaler.fit_transform(
        df[['popularity', 'vote_average']]
    )

    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # Save processed CSV
    out_path = os.path.join(DATA_DIR, 'processed_movies.csv')
    df.to_csv(out_path, index=False)
    print(f"\nProcessed dataset saved to {out_path}")
    print(f"Mood distribution:\n{df['mood_label'].value_counts()}")

    return df, all_genres


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – VISUALISATION: Genre Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_genre_distribution(df, all_genres):
    print("\nPlotting genre distribution…")
    genre_counts = {g: df[f'genre_{g}'].sum() for g in all_genres}
    genre_counts = dict(sorted(genre_counts.items(), key=lambda x: -x[1])[:15])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE['cream'])
    ax.set_facecolor(PALETTE['cream'])

    bars = ax.barh(list(genre_counts.keys()), list(genre_counts.values()),
                   color=PALETTE['burgundy'], edgecolor=PALETTE['terracotta'],
                   linewidth=0.8)

    # Gradient-style tint
    for i, bar in enumerate(bars):
        alpha = 0.6 + 0.4 * (i / len(bars))
        bar.set_alpha(alpha)

    ax.set_xlabel('Number of Movies', fontsize=12, color=PALETTE['charcoal'])
    ax.set_title('Top 15 Genres in TMDb Dataset', fontsize=16,
                 fontweight='bold', color=PALETTE['charcoal'], pad=16)
    ax.tick_params(colors=PALETTE['charcoal'])
    for spine in ax.spines.values():
        spine.set_color(PALETTE['terracotta'])
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'genre_distribution.png'), dpi=150)
    plt.close()
    print("  -> genre_distribution.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – FEATURE ENGINEERING: PCA
# ══════════════════════════════════════════════════════════════════════════════

def apply_pca(df, all_genres):
    print("\n" + "="*60)
    print("STEP 3: Feature Engineering – PCA")
    print("="*60)

    genre_cols = [f'genre_{g}' for g in all_genres]
    feature_cols = genre_cols + ['popularity_scaled', 'vote_scaled']
    X = df[feature_cols].values

    # Determine how many components explain ≥ 90% variance
    pca_full = PCA()
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= 0.90)) + 1
    n_components = max(n_components, 5)   # at least 5
    print(f"Components selected: {n_components}  (explains {cumvar[n_components-1]*100:.1f}% variance)")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_model.pkl'))

    # ── PCA variance plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE['cream'])

    for ax in axes:
        ax.set_facecolor(PALETTE['cream'])
        for spine in ax.spines.values():
            spine.set_color(PALETTE['terracotta'])
            spine.set_linewidth(0.5)

    # Individual explained variance
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_[:20]) + 1),
                pca_full.explained_variance_ratio_[:20],
                color=PALETTE['burgundy'], alpha=0.85)
    axes[0].set_title('Explained Variance per Component', fontsize=13,
                      fontweight='bold', color=PALETTE['charcoal'])
    axes[0].set_xlabel('Component', color=PALETTE['charcoal'])
    axes[0].set_ylabel('Explained Variance Ratio', color=PALETTE['charcoal'])
    axes[0].tick_params(colors=PALETTE['charcoal'])

    # Cumulative
    axes[1].plot(range(1, len(cumvar) + 1), cumvar,
                 color=PALETTE['burgundy'], linewidth=2.5)
    axes[1].axhline(0.90, color=PALETTE['terracotta'],
                    linestyle='--', linewidth=1.5, label='90% threshold')
    axes[1].axvline(n_components, color=PALETTE['gold'],
                    linestyle='--', linewidth=1.5, label=f'n={n_components}')
    axes[1].set_title('Cumulative Explained Variance', fontsize=13,
                      fontweight='bold', color=PALETTE['charcoal'])
    axes[1].set_xlabel('Number of Components', color=PALETTE['charcoal'])
    axes[1].set_ylabel('Cumulative Variance', color=PALETTE['charcoal'])
    axes[1].tick_params(colors=PALETTE['charcoal'])
    axes[1].legend(fontsize=10)

    plt.suptitle('PCA Analysis', fontsize=16, fontweight='bold',
                 color=PALETTE['charcoal'], y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'pca_variance.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  -> pca_variance.png saved")

    return X_pca, feature_cols, pca


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def train_classifiers(X_pca, df):
    print("\n" + "="*60)
    print("STEP 4: Training Classifiers")
    print("="*60)

    le = LabelEncoder()
    y = le.fit_transform(df['mood_label'])
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    models = {
        'SVM': SVC(kernel='rbf', C=5.0, gamma='scale',
                   probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}…")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'model'    : model,
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall'   : recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1'       : f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cm'       : confusion_matrix(y_test, y_pred),
            'y_pred'   : y_pred,
            'y_test'   : y_test,
        }
        print(f"    Accuracy : {results[name]['accuracy']:.4f}")
        print(f"    F1 Score : {results[name]['f1']:.4f}")

    # Save best model (SVM is primary; also save others)
    joblib.dump(results['SVM']['model'],
                os.path.join(MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(results['KNN']['model'],
                os.path.join(MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(results['Logistic Regression']['model'],
                os.path.join(MODEL_DIR, 'lr_model.pkl'))

    print("\n  -- Model Comparison -------------------------")
    print(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*52}")
    for name, r in results.items():
        print(f"  {name:<22} {r['accuracy']:>7.4f} {r['precision']:>7.4f}"
              f" {r['recall']:>7.4f} {r['f1']:>7.4f}")

    best = max(results, key=lambda n: results[n]['f1'])
    print(f"\n  * Best model: {best}  (F1={results[best]['f1']:.4f})")

    return results, le, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – PLOT CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(results, le):
    print("\nPlotting confusion matrices…")
    class_names = le.classes_

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(PALETTE['cream'])

    cmap = sns.light_palette(PALETTE['burgundy'], as_cmap=True)

    for ax, (name, r) in zip(axes, results.items()):
        cm = r['cm']
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=False, linewidths=0.5,
                    linecolor=PALETTE['cream'])
        ax.set_facecolor(PALETTE['cream'])
        ax.set_title(f'{name}\nAcc={r["accuracy"]:.3f}  F1={r["f1"]:.3f}',
                     fontsize=11, fontweight='bold', color=PALETTE['charcoal'])
        ax.set_xlabel('Predicted', color=PALETTE['charcoal'])
        ax.set_ylabel('Actual', color=PALETTE['charcoal'])
        ax.tick_params(colors=PALETTE['charcoal'], rotation=30)

    plt.suptitle('Confusion Matrices (Normalised)', fontsize=15,
                 fontweight='bold', color=PALETTE['charcoal'])
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrices.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  -> confusion_matrices.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def apply_clustering(X_pca, df):
    print("\n" + "="*60)
    print("STEP 6: K-Means Clustering")
    print("="*60)

    # Elbow method
    inertias = []
    K_range  = range(2, 13)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)

    # Plot elbow
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE['cream'])
    ax.set_facecolor(PALETTE['cream'])

    ax.plot(list(K_range), inertias, 'o-',
            color=PALETTE['burgundy'], linewidth=2.5, markersize=7,
            markerfacecolor=PALETTE['terracotta'])
    ax.set_xlabel('Number of Clusters (K)', fontsize=12, color=PALETTE['charcoal'])
    ax.set_ylabel('Inertia (WCSS)', fontsize=12, color=PALETTE['charcoal'])
    ax.set_title('Elbow Method – Optimal K for K-Means', fontsize=14,
                 fontweight='bold', color=PALETTE['charcoal'])
    ax.tick_params(colors=PALETTE['charcoal'])
    for spine in ax.spines.values():
        spine.set_color(PALETTE['terracotta'])
        spine.set_linewidth(0.5)
    ax.axvline(5, color=PALETTE['gold'], linestyle='--',
               linewidth=1.5, label='K=5 (chosen)')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'elbow_method.png'), dpi=150)
    plt.close()
    print("  -> elbow_method.png saved")

    # Train final K-Means with K=5
    K_FINAL = 5
    kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    df = df.copy()
    df['cluster'] = cluster_labels
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    print(f"  K-Means with K={K_FINAL} trained and saved")
    print(f"  Cluster distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

    return df, kmeans


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – SAVE METADATA FOR BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def save_metadata(df, all_genres, results, le):
    print("\nSaving metadata…")

    # Model comparison metrics
    metrics = {
        name: {
            'accuracy' : float(r['accuracy']),
            'precision': float(r['precision']),
            'recall'   : float(r['recall']),
            'f1'       : float(r['f1']),
        }
        for name, r in results.items()
    }

    meta = {
        'all_genres'   : all_genres,
        'mood_labels'  : list(le.classes_),
        'model_metrics': metrics,
        'best_model'   : max(metrics, key=lambda n: metrics[n]['f1']),
    }

    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("  -> metadata.json saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df, all_genres = load_and_preprocess()
    plot_genre_distribution(df, all_genres)
    X_pca, feature_cols, pca = apply_pca(df, all_genres)
    results, le, X_train, X_test, y_train, y_test = train_classifiers(X_pca, df)
    plot_confusion_matrices(results, le)
    df, kmeans = apply_clustering(X_pca, df)
    save_metadata(df, all_genres, results, le)

    # Save enriched processed dataset (with cluster column)
    df.to_csv(os.path.join(DATA_DIR, 'processed_movies.csv'), index=False)

    print("\n" + "="*60)
    print("Training complete! All models & artefacts saved.")
    print("="*60)
