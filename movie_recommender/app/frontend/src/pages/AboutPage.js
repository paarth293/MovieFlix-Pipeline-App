import React from 'react';
import './AboutPage.css';

const TECH_STACK = [
  { layer: 'Dataset',         tech: 'TMDb 5000 Movies',        icon: '🎬' },
  { layer: 'Preprocessing',   tech: 'Pandas · StandardScaler', icon: '🧹' },
  { layer: 'Feature Eng.',    tech: 'PCA (scikit-learn)',       icon: '📐' },
  { layer: 'Classification',  tech: 'SVM · KNN · Logistic Reg',icon: '🤖' },
  { layer: 'Clustering',      tech: 'K-Means (K=5)',           icon: '🔍' },
  { layer: 'Serialisation',   tech: 'joblib (.pkl files)',      icon: '💾' },
  { layer: 'Backend API',     tech: 'Flask · Flask-CORS',      icon: '⚙️'  },
  { layer: 'Frontend',        tech: 'React 18 · Recharts',     icon: '🖥️'  },
];

const FILE_TREE = `movie_recommender/
├── data/
│   ├── tmdb_5000_movies.csv       # raw dataset
│   └── processed_movies.csv       # cleaned + encoded + mood labels
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
│   │   ├── app.py                  # Flask API
│   │   └── recommender.py          # core ML logic
│   └── frontend/
│       ├── public/index.html
│       ├── package.json
│       └── src/
│           ├── components/         # Navbar, MovieCard, MoodSelector …
│           ├── pages/              # Home, Recommend, Metrics, About
│           └── hooks/useApi.js
├── train.py                        # end-to-end training pipeline
├── requirements.txt
└── README.md`;

export default function AboutPage() {
  return (
    <div className="about-page">
      {/* Hero */}
      <div className="about-hero">
        <div className="section">
          <p className="about-hero__eyebrow">Academic ML Project</p>
          <h1 className="about-hero__title">About MoodFlix</h1>
          <p className="about-hero__sub">
            An end-to-end emotion-based movie recommendation system built with
            scikit-learn, Flask, and React.
          </p>
        </div>
      </div>

      <div className="section about-body">
        {/* Overview */}
        <div className="about-card">
          <h2 className="about-section-title">Project Overview</h2>
          <p className="about-text">
            MoodFlix demonstrates a complete ML pipeline from raw data to a
            deployed recommendation interface. The user's mood is converted into
            a feature vector, classified by an SVM model trained on TMDb genre
            data, then refined by K-Means clustering to surface the most relevant
            films.
          </p>
          <p className="about-text">
            The project covers every core academic ML component: data
            preprocessing, feature engineering (PCA), multi-class classification
            with three models (SVM, KNN, Logistic Regression), unsupervised
            clustering, and a full evaluation suite with accuracy, precision,
            recall, F1, and confusion matrices.
          </p>
        </div>

        {/* Tech stack */}
        <div className="about-card">
          <h2 className="about-section-title">Technology Stack</h2>
          <div className="tech-grid">
            {TECH_STACK.map((t) => (
              <div key={t.layer} className="tech-row">
                <span className="tech-icon">{t.icon}</span>
                <div>
                  <div className="tech-layer">{t.layer}</div>
                  <div className="tech-name">{t.tech}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* File tree */}
        <div className="about-card">
          <h2 className="about-section-title">Project Structure</h2>
          <pre className="file-tree">{FILE_TREE}</pre>
        </div>

        {/* Run instructions */}
        <div className="about-card">
          <h2 className="about-section-title">How to Run</h2>

          <div className="run-steps">
            {[
              {
                n: '1',
                title: 'Install Python dependencies',
                code: 'pip install -r requirements.txt',
              },
              {
                n: '2',
                title: 'Train the ML pipeline',
                code: 'python train.py',
                note: 'Generates models/ and visualizations/ artefacts',
              },
              {
                n: '3',
                title: 'Start the Flask API',
                code: 'cd app/backend && python app.py',
                note: 'Runs on http://localhost:5000',
              },
              {
                n: '4',
                title: 'Start the React frontend',
                code: 'cd app/frontend && npm install && npm start',
                note: 'Opens on http://localhost:3000',
              },
            ].map((s) => (
              <div key={s.n} className="run-step">
                <div className="run-step__num">{s.n}</div>
                <div className="run-step__content">
                  <p className="run-step__title">{s.title}</p>
                  <code className="run-step__code">{s.code}</code>
                  {s.note && <p className="run-step__note">{s.note}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Mood mapping */}
        <div className="about-card">
          <h2 className="about-section-title">Mood → Genre Mapping</h2>
          <div className="mapping-grid">
            {[
              { mood: '😄 Happy',   genre: 'Comedy',  color: '#C4663A' },
              { mood: '😢 Sad',     genre: 'Drama',   color: '#3A5A8C' },
              { mood: '😤 Angry',   genre: 'Action',  color: '#8B2D1A' },
              { mood: '😌 Relaxed', genre: 'Romance', color: '#3A6B3E' },
              { mood: '😐 Neutral', genre: 'Mixed',   color: '#5A4A6B' },
            ].map((m) => (
              <div
                key={m.mood}
                className="mapping-item"
                style={{ borderLeftColor: m.color }}
              >
                <span className="mapping-mood">{m.mood}</span>
                <span className="mapping-arrow">→</span>
                <span className="mapping-genre" style={{ color: m.color }}>
                  {m.genre}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
