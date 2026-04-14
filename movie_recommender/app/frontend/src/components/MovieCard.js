import React, { useState } from 'react';
import './MovieCard.css';

const MOOD_COLORS = {
  Happy:   { bg: '#FFF3DC', accent: '#C4663A', dot: '#D4AF37' },
  Sad:     { bg: '#EDF0F7', accent: '#3A5A8C', dot: '#6A8FBD' },
  Angry:   { bg: '#FBE9E7', accent: '#8B2D1A', dot: '#C4663A' },
  Relaxed: { bg: '#EBF3EC', accent: '#3A6B3E', dot: '#7A9E7E' },
  Neutral: { bg: '#F5F0F8', accent: '#5A4A6B', dot: '#8A7A9E' },
};

function StarRating({ rating }) {
  const max = 10;
  const filled = Math.round((rating / max) * 5);
  return (
    <div className="star-rating" title={`${rating}/10`}>
      {[1,2,3,4,5].map(i => (
        <span key={i} className={i <= filled ? 'star star--on' : 'star star--off'}>
          ★
        </span>
      ))}
      <span className="rating-num">{rating}/10</span>
    </div>
  );
}

export default function MovieCard({ movie, index }) {
  const [expanded, setExpanded] = useState(false);
  const colors = MOOD_COLORS[movie.mood] || MOOD_COLORS.Neutral;

  return (
    <article
      className="movie-card"
      style={{ '--accent': colors.accent, '--card-bg': colors.bg, animationDelay: `${index * 0.08}s` }}
    >
      {/* Rank badge */}
      <div className="card-rank">#{index + 1}</div>

      {/* Mood dot */}
      <div className="card-mood-dot" style={{ background: colors.dot }} />

      <div className="card-body">
        <h3 className="card-title">{movie.title}</h3>

        {/* Genres */}
        <div className="card-genres">
          {movie.genres.map(g => (
            <span key={g} className="genre-chip">{g}</span>
          ))}
        </div>

        <StarRating rating={movie.rating} />

        {/* Overview */}
        <p className={`card-overview ${expanded ? 'card-overview--expanded' : ''}`}>
          {movie.overview || 'No overview available.'}
        </p>
        {movie.overview && movie.overview.length > 120 && (
          <button
            className="card-toggle"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Show less ↑' : 'Read more ↓'}
          </button>
        )}
      </div>

      {/* Footer metrics */}
      <div className="card-footer">
        <span className="card-meta">
          <span className="meta-label">Mood</span>
          <span className="meta-value">{movie.mood}</span>
        </span>
        <span className="card-meta">
          <span className="meta-label">Popularity</span>
          <span className="meta-value">{movie.popularity}</span>
        </span>
        <span className="card-meta">
          <span className="meta-label">Cluster</span>
          <span className="meta-value">#{movie.cluster}</span>
        </span>
      </div>
    </article>
  );
}
