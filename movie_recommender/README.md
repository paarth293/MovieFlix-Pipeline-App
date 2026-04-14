# 🎬 MoodFlix — Emotion-Based Movie Recommendation System

An end-to-end machine learning project that recommends movies based on your current mood using **SVM classification** and **K-Means clustering**, built with Python (scikit-learn) + Flask + React.

---

## 📊 Results at a Glance

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| **SVM** ⭐ Best     | 92.88%   | 92.87%    | 92.88% | 92.44%   |
| KNN                 | 91.41%   | 91.10%    | 91.41% | 90.88%   |
| Logistic Regression | 92.88%   | 93.11%    | 92.88% | 92.41%   |

- **Dataset**: TMDb 5000 Movies (4,775 usable rows after cleaning)
- **PCA Components**: 11 (explain 90.9% variance)
- **K-Means Clusters**: K=5 (Elbow method)

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
cd movie_recommender
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML pipeline

```bash
python train.py
```

This will:
- Preprocess `data/tmdb_5000_movies.csv`
- Save `data/processed_movies.csv`
- Train SVM, KNN, Logistic Regression classifiers
- Train K-Means clustering model
- Save all `.pkl` model files to `models/`
- Generate 4 visualisation plots in `visualizations/`

### 4. Start the Flask API

```bash
cd app/backend
python app.py
```

API runs on **http://localhost:5000**

### 5. Start the React frontend

```bash
cd app/frontend
npm install
npm start
```

Frontend opens on **http://localhost:3000**

---

## 🌐 API Endpoints

| Method | Endpoint         | Description                         |
|--------|-----------------|-------------------------------------|
| GET    | `/api/health`   | Health check                        |
| POST   | `/api/recommend`| Get movie recommendations           |
| GET    | `/api/metrics`  | Model evaluation metrics            |
| GET    | `/api/moods`    | Available moods with descriptions   |
| GET    | `/api/metadata` | System metadata                     |

### POST `/api/recommend`

```json
{
  "mood": "Happy",
  "model": "svm",
  "top_n": 5
}
```

Response:
```json
{
  "mood": "Happy",
  "predicted_label": "Happy",
  "model_used": "SVM",
  "movies": [
    {
      "title": "The Grand Budapest Hotel",
      "genres": ["Comedy", "Drama"],
      "rating": 7.9,
      "popularity": 52.3,
      "overview": "...",
      "mood": "Happy",
      "cluster": 1
    }
  ]
}
```

---

## 📐 ML Pipeline

```
Raw CSV
  → Parse JSON genres → mood label mapping
  → One-hot encode genres (20 columns)
  → StandardScaler on popularity & vote_average
  → PCA (11 components, 90.9% variance)
  → Train SVM (primary) / KNN / Logistic Regression
  → K-Means clustering (K=5)
  → Save .pkl artefacts
  → Flask API serves predictions
  → React UI presents results
```

### Mood ↔ Genre Mapping

| Mood    | Primary Genre |
|---------|--------------|
| Happy   | Comedy       |
| Sad     | Drama        |
| Angry   | Action       |
| Relaxed | Romance      |
| Neutral | (Mixed)      |

---

## 📁 Project Structure

```
movie_recommender/
├── data/
│   ├── tmdb_5000_movies.csv
│   └── processed_movies.csv
├── models/
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── pca_model.pkl
│   ├── kmeans_model.pkl
│   ├── label_encoder.pkl
│   └── metadata.json
├── visualizations/
│   ├── genre_distribution.png
│   ├── pca_variance.png
│   ├── confusion_matrices.png
│   └── elbow_method.png
├── app/
│   ├── backend/
│   │   ├── app.py
│   │   └── recommender.py
│   └── frontend/
│       ├── public/
│       ├── package.json
│       └── src/
│           ├── components/
│           ├── pages/
│           └── hooks/
├── train.py
├── requirements.txt
└── README.md
```

---

## 🎓 Academic Components Demonstrated

- ✅ Data preprocessing (missing values, JSON parsing, encoding, scaling)
- ✅ Feature engineering (PCA with variance analysis)
- ✅ Classification — SVM (primary), KNN, Logistic Regression
- ✅ Model evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix
- ✅ Unsupervised clustering — K-Means with Elbow Method
- ✅ Recommendation logic combining classification + clustering
- ✅ Visualisations — genre dist., PCA variance, confusion matrices, elbow plot
- ✅ REST API backend (Flask)
- ✅ Interactive UI frontend (React)
