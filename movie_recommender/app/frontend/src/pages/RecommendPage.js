import React, { useState } from 'react';
import MoodSelector from '../components/MoodSelector';
import MovieCard from '../components/MovieCard';
import { useRecommend } from '../hooks/useApi';
import './RecommendPage.css';

const MODEL_OPTIONS = [
  { value: 'svm', label: 'SVM',                desc: 'Primary · Best accuracy' },
  { value: 'knn', label: 'KNN',                desc: 'K-Nearest Neighbours' },
  { value: 'lr',  label: 'Logistic Regression', desc: 'Fast linear baseline' },
];

const COUNT_OPTIONS = [3, 5, 8, 10];

export default function RecommendPage() {
  const [mood,      setMood]      = useState('Happy');
  const [model,     setModel]     = useState('svm');
  const [topN,      setTopN]      = useState(5);
  const { movies, label, loading, error, fetchRecs } = useRecommend();

  const handleSubmit = () => fetchRecs(mood, model, topN);

  return (
    <div className="recommend-page">
      {/* ── Page header ─────────────────────────────────────────────────── */}
      <div className="rec-hero">
        <div className="section">
          <p className="rec-hero__eyebrow">Powered by SVM + K-Means</p>
          <h1 className="rec-hero__title">How are you feeling?</h1>
          <p className="rec-hero__sub">
            Choose your current mood and let our ML pipeline find the perfect film.
          </p>
        </div>
      </div>

      <div className="section rec-body">
        {/* ── Mood selector ─────────────────────────────────────────────── */}
        <div className="rec-panel">
          <h2 className="panel-title">
            <span className="panel-num">01</span> Select Your Mood
          </h2>
          <MoodSelector selected={mood} onSelect={setMood} />
        </div>

        {/* ── Options ───────────────────────────────────────────────────── */}
        <div className="rec-options">
          {/* Model picker */}
          <div className="rec-panel rec-panel--half">
            <h2 className="panel-title">
              <span className="panel-num">02</span> ML Model
            </h2>
            <div className="model-options">
              {MODEL_OPTIONS.map((m) => (
                <button
                  key={m.value}
                  className={`model-btn ${model === m.value ? 'model-btn--active' : ''}`}
                  onClick={() => setModel(m.value)}
                >
                  <span className="model-btn__label">{m.label}</span>
                  <span className="model-btn__desc">{m.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Count picker */}
          <div className="rec-panel rec-panel--half">
            <h2 className="panel-title">
              <span className="panel-num">03</span> Number of Results
            </h2>
            <div className="count-options">
              {COUNT_OPTIONS.map((n) => (
                <button
                  key={n}
                  className={`count-btn ${topN === n ? 'count-btn--active' : ''}`}
                  onClick={() => setTopN(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Submit ────────────────────────────────────────────────────── */}
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? (
            <span className="submit-btn__loading">
              <span className="spinner" />
              Analysing…
            </span>
          ) : (
            `Get Recommendations for "${mood}" →`
          )}
        </button>

        {/* ── Error ─────────────────────────────────────────────────────── */}
        {error && (
          <div className="rec-error">
            ⚠️ {error}
          </div>
        )}

        {/* ── Results ───────────────────────────────────────────────────── */}
        {movies.length > 0 && (
          <div className="results-section">
            <div className="results-header">
              <h2 className="results-title">
                Top {movies.length} films for <em>{mood}</em>
              </h2>
              <div className="results-meta">
                <span className="results-tag">Model: {model.toUpperCase()}</span>
                <span className="results-tag">Predicted: {label}</span>
              </div>
            </div>

            <div className="movies-grid">
              {movies.map((movie, i) => (
                <MovieCard key={movie.title + i} movie={movie} index={i} />
              ))}
            </div>
          </div>
        )}

        {/* ── Empty state ───────────────────────────────────────────────── */}
        {!loading && movies.length === 0 && !error && (
          <div className="empty-state">
            <span className="empty-icon">🎞️</span>
            <p>Select a mood and click <strong>Get Recommendations</strong> to begin.</p>
          </div>
        )}
      </div>
    </div>
  );
}
