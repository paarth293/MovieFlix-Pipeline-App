import React, { useEffect } from 'react';
import { ModelRadarChart, MetricsBarChart, MetricsTable } from '../components/MetricsChart';
import { useMetrics } from '../hooks/useApi';
import './MetricsPage.css';

const ML_STEPS = [
  {
    icon: '🧹',
    title: 'Data Preprocessing',
    items: [
      'Parsed JSON genre column → list of genre names',
      'Dropped 3 missing overviews, removed genre-less rows',
      'One-hot encoded 20 genres',
      'StandardScaler on popularity & vote_average',
      'Mapped genres → mood labels (Comedy→Happy, Drama→Sad, Action→Angry, Romance→Relaxed)',
    ],
  },
  {
    icon: '📐',
    title: 'PCA Feature Engineering',
    items: [
      '22 features (20 genre flags + 2 scaled numerics)',
      '11 principal components explain 90.9% variance',
      'Reduces noise before classification',
      'Speeds up SVM inference significantly',
    ],
  },
  {
    icon: '🤖',
    title: 'SVM Classification (Primary)',
    items: [
      'RBF kernel, C=5.0, gamma=scale',
      '80/20 train-test split, stratified',
      'Achieves 92.88% accuracy on held-out test set',
      'Probability estimates enabled for future ranking',
    ],
  },
  {
    icon: '🔍',
    title: 'K-Means Clustering',
    items: [
      'Elbow method tested K = 2…12',
      'K=5 chosen as optimal inflection point',
      'Cluster assignment refines mood-filtered results',
      'Query vector projected into cluster space for matching',
    ],
  },
];

export default function MetricsPage() {
  const { metrics, loading, error, fetchMetrics } = useMetrics();

  useEffect(() => { fetchMetrics(); }, [fetchMetrics]);

  return (
    <div className="metrics-page">
      {/* Hero */}
      <div className="metrics-hero">
        <div className="section">
          <p className="metrics-hero__eyebrow">Evaluation · Results · Pipeline</p>
          <h1 className="metrics-hero__title">ML Model Performance</h1>
          <p className="metrics-hero__sub">
            Detailed evaluation of SVM, KNN and Logistic Regression classifiers trained
            on the TMDb dataset with PCA-reduced features.
          </p>
        </div>
      </div>

      <div className="section metrics-body">
        {/* ── Charts ──────────────────────────────────────────────────────── */}
        {loading && (
          <div className="metrics-loading">
            <div className="spinner" />
            <p>Loading model metrics…</p>
          </div>
        )}

        {error && (
          <div className="metrics-error">
            ⚠️ Could not load metrics. Make sure the Flask API is running on port 5000.
            <code>{error}</code>
          </div>
        )}

        {metrics && (
          <>
            <div className="charts-grid">
              <ModelRadarChart metrics={metrics} />
              <MetricsBarChart metrics={metrics} />
            </div>
            <MetricsTable metrics={metrics} />
          </>
        )}

        {/* ── Pipeline explanation ────────────────────────────────────────── */}
        <div className="pipeline-section">
          <h2 className="pipeline-title">ML Pipeline Walkthrough</h2>
          <div className="pipeline-grid">
            {ML_STEPS.map((step, i) => (
              <div key={step.title} className="pipeline-card" style={{ animationDelay: `${i * 0.08}s` }}>
                <div className="pipeline-card__header">
                  <span className="pipeline-icon">{step.icon}</span>
                  <h3 className="pipeline-card__title">{step.title}</h3>
                </div>
                <ul className="pipeline-list">
                  {step.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* ── Mood label distribution ─────────────────────────────────────── */}
        <div className="distribution-section">
          <h2 className="pipeline-title">Mood Label Distribution</h2>
          <p className="pipeline-sub">
            Labels derived from primary genre mapping on 4,775 cleaned movies.
          </p>
          <div className="mood-dist-grid">
            {[
              { mood: 'Sad',     n: 1602, pct: 33.6, color: '#3A5A8C' },
              { mood: 'Happy',   n: 1299, pct: 27.2, color: '#C4663A' },
              { mood: 'Angry',   n: 1034, pct: 21.7, color: '#8B2D1A' },
              { mood: 'Neutral', n: 718,  pct: 15.0, color: '#5A4A6B' },
              { mood: 'Relaxed', n: 122,  pct: 2.6,  color: '#3A6B3E' },
            ].map((d) => (
              <div key={d.mood} className="mood-dist-item">
                <div className="mood-dist-top">
                  <span className="mood-dist-name">{d.mood}</span>
                  <span className="mood-dist-count">{d.n.toLocaleString()}</span>
                </div>
                <div className="mood-dist-bar">
                  <div
                    className="mood-dist-fill"
                    style={{ width: `${d.pct}%`, background: d.color }}
                  />
                </div>
                <span className="mood-dist-pct">{d.pct}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
