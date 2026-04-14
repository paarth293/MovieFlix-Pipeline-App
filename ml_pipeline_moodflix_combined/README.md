# MoodFlix ✦ ML Pipeline Lab

A cinematic fusion of two ML projects:
1. **MoodFlix** — Emotion-based movie recommender (SVM + K-Means + PCA on TMDb 5000)
2. **ML Pipeline Lab** — Interactive 9-step ML workflow that works on any CSV dataset

---

## Setup

```bash
pip install -r requirements.txt

# If using MoodFlix recommendations, train models first:
cd movie_recommender
python train.py
cd ..

# Run the app
streamlit run ml_pipeline_app.py
```

Place this file alongside the `movie_recommender/` folder from your original project.

---

## MoodFlix Features
- 5 mood categories → SVM classification → K-Means cluster refinement
- 3 model comparison: SVM, KNN, Logistic Regression
- Cinematic UI with Playfair Display typography

## ML Pipeline Lab — 9 Steps

| Step | Name | What it does |
|------|------|-------------|
| 1 | Problem Type | Classification or Regression |
| 2 | Data Input | CSV upload, target selection, PCA visualisation |
| 3 | EDA | Distributions, correlations, missing values |
| 4 | Data Engineering | Mean/Median/Mode imputation + IQR/IsolationForest/DBSCAN/OPTICS outliers |
| 5 | Feature Selection | Variance threshold, correlation filter, mutual info |
| 6 | Data Split | Train/test split with optional stratification |
| 7 | Model Selection | SVM (kernel options), Random Forest, KNN, LogReg, Linear/Ridge |
| 8 | Training + KFold | Cross-validation with per-fold visualisation |
| 9 | Performance Metrics | Accuracy/R², confusion matrix, overfit/underfit detection |
| 10 | Hyperparameter Tuning | GridSearchCV or RandomizedSearchCV with before/after comparison |
