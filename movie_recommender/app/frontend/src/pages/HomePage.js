import React from 'react';
import './HomePage.css';

const FEATURES = [
  {
    icon: '🧠',
    title: 'SVM Classification',
    body: 'A Support Vector Machine with RBF kernel achieves 92.88% accuracy in mapping your mood to the right film genre category.',
  },
  {
    icon: '🔍',
    title: 'K-Means Clustering',
    body: 'Movies are grouped into 5 clusters based on genre and quality signals. Cluster matching refines results beyond simple label filtering.',
  },
  {
    icon: '📐',
    title: 'PCA Dimensionality Reduction',
    body: '11 principal components capture 90.9% of variance in 20+ genre features, making inference fast and reducing noise.',
  },
  {
    icon: '🎬',
    title: '4,700+ Movies',
    body: 'The full TMDb 5000 dataset, cleaned and enriched with mood labels, one-hot encoded genres, and normalised popularity scores.',
  },
];

const STATS = [
  { value: '92.88%', label: 'SVM Accuracy' },
  { value: '4,775',  label: 'Movies in dataset' },
  { value: '11',     label: 'PCA Components' },
  { value: '5',      label: 'K-Means Clusters' },
];

export default function HomePage({ navigate }) {
  return (
    <div className="home">
      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero__bg-text" aria-hidden>CINEMA</div>
        <div className="hero__inner section">
          <div className="hero__eyebrow">Machine Learning · TMDb Dataset</div>
          <h1 className="hero__title">
            Your Mood,<br />
            <em>Your Movie.</em>
          </h1>
          <p className="hero__desc">
            Tell us how you feel. Our SVM classifier and K-Means clustering
            pipeline will surface the films that match your emotional state —
            curated from 4,700+ titles using real ML.
          </p>
          <div className="hero__actions">
            <button className="btn-primary" onClick={() => navigate('recommend')}>
              Find My Film →
            </button>
            <button className="btn-ghost" onClick={() => navigate('metrics')}>
              View ML Metrics
            </button>
          </div>
        </div>
      </section>

      {/* ── Stats bar ─────────────────────────────────────────────────────── */}
      <section className="stats-bar">
        <div className="section stats-bar__inner">
          {STATS.map((s) => (
            <div key={s.label} className="stat-item">
              <span className="stat-value">{s.value}</span>
              <span className="stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────────────────── */}
      <section className="how-it-works section">
        <div className="section-header">
          <span className="section-eyebrow">The Pipeline</span>
          <h2 className="section-title">How MoodFlix Works</h2>
        </div>

        <div className="steps">
          {[
            { n: '01', title: 'Pick a Mood', body: 'Choose from Happy, Sad, Angry, Relaxed, or Neutral.' },
            { n: '02', title: 'SVM Predicts', body: 'Your input is transformed via PCA and fed to the trained SVM model.' },
            { n: '03', title: 'Cluster Refines', body: 'K-Means pinpoints the best-matching cluster of movies.' },
            { n: '04', title: 'Top 5 Served', body: 'Results sorted by rating, with genre and overview details.' },
          ].map((s) => (
            <div key={s.n} className="step">
              <div className="step-num">{s.n}</div>
              <div className="step-content">
                <h3 className="step-title">{s.title}</h3>
                <p className="step-body">{s.body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature cards ────────────────────────────────────────────────── */}
      <section className="features">
        <div className="section">
          <div className="section-header">
            <span className="section-eyebrow">Under the Hood</span>
            <h2 className="section-title">ML Architecture</h2>
          </div>
          <div className="feature-grid">
            {FEATURES.map((f) => (
              <div key={f.title} className="feature-card">
                <span className="feature-icon">{f.icon}</span>
                <h3 className="feature-title">{f.title}</h3>
                <p className="feature-body">{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA band ─────────────────────────────────────────────────────── */}
      <section className="cta-band">
        <div className="section cta-band__inner">
          <h2 className="cta-band__title">Ready to find tonight's film?</h2>
          <p className="cta-band__sub">Takes two seconds. No sign-up required.</p>
          <button className="btn-primary btn-primary--light" onClick={() => navigate('recommend')}>
            Get Recommendations →
          </button>
        </div>
      </section>
    </div>
  );
}
