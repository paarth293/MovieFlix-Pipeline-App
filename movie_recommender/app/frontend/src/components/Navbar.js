import React, { useEffect, useState } from 'react';
import './Navbar.css';

const NAV_ITEMS = [
  { id: 'home',      label: 'Home' },
  { id: 'recommend', label: 'Get Recs' },
  { id: 'metrics',   label: 'ML Metrics' },
  { id: 'about',     label: 'About' },
];

export default function Navbar({ current, navigate }) {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handler);
    return () => window.removeEventListener('scroll', handler);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
      <div className="navbar__inner">
        <button className="navbar__brand" onClick={() => navigate('home')}>
          <span className="brand-icon">🎬</span>
          <span className="brand-name">MoodFlix</span>
        </button>

        {/* Desktop nav */}
        <ul className="navbar__links">
          {NAV_ITEMS.map((item) => (
            <li key={item.id}>
              <button
                className={`nav-link ${current === item.id ? 'nav-link--active' : ''}`}
                onClick={() => navigate(item.id)}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>

        {/* CTA button */}
        <button
          className="navbar__cta"
          onClick={() => navigate('recommend')}
        >
          Find My Film
        </button>

        {/* Mobile hamburger */}
        <button
          className={`navbar__burger ${menuOpen ? 'open' : ''}`}
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span /><span /><span />
        </button>
      </div>

      {/* Mobile drawer */}
      {menuOpen && (
        <div className="navbar__mobile">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`mobile-link ${current === item.id ? 'mobile-link--active' : ''}`}
              onClick={() => { navigate(item.id); setMenuOpen(false); }}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </nav>
  );
}
