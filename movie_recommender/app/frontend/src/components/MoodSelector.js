import React from 'react';
import './MoodSelector.css';

const MOODS = [
  {
    label: 'Happy',
    emoji: '😄',
    subtitle: 'Comedy · Feel-good',
    color: '#C4663A',
    bg: '#FFF3DC',
  },
  {
    label: 'Sad',
    emoji: '😢',
    subtitle: 'Drama · Emotional',
    color: '#3A5A8C',
    bg: '#EDF0F7',
  },
  {
    label: 'Angry',
    emoji: '😤',
    subtitle: 'Action · Adrenaline',
    color: '#8B2D1A',
    bg: '#FBE9E7',
  },
  {
    label: 'Relaxed',
    emoji: '😌',
    subtitle: 'Romance · Heartwarming',
    color: '#3A6B3E',
    bg: '#EBF3EC',
  },
  {
    label: 'Neutral',
    emoji: '😐',
    subtitle: 'Mixed · Any genre',
    color: '#5A4A6B',
    bg: '#F5F0F8',
  },
];

export default function MoodSelector({ selected, onSelect }) {
  return (
    <div className="mood-selector">
      {MOODS.map((mood) => (
        <button
          key={mood.label}
          className={`mood-tile ${selected === mood.label ? 'mood-tile--active' : ''}`}
          style={{
            '--mood-color': mood.color,
            '--mood-bg': mood.bg,
          }}
          onClick={() => onSelect(mood.label)}
          aria-pressed={selected === mood.label}
        >
          <span className="mood-emoji">{mood.emoji}</span>
          <span className="mood-label">{mood.label}</span>
          <span className="mood-subtitle">{mood.subtitle}</span>
          {selected === mood.label && (
            <span className="mood-check">✓</span>
          )}
        </button>
      ))}
    </div>
  );
}
